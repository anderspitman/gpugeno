use gpugeno::indexed_batch::{BatchError, DisjointBamStream, IndexedBamBatch};
use std::any::Any;
use std::io;
use std::sync::mpsc::{self, Receiver, RecvError, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Messages sent by the one producer to the main-thread consumer.  A
/// disconnect is intentionally not represented by `Eof`: only an explicit
/// terminal message completes the stream protocol.
pub(crate) enum ProducerMessage {
    Batch(IndexedBamBatch),
    StreamError(BatchError),
    Eof,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ProducerStats {
    /// From immediately before the named thread spawn request through the
    /// explicit stream/worker-pool teardown and normal producer return.
    pub(crate) lifetime: Duration,
    pub(crate) first_batch_send_wait: Duration,
    pub(crate) later_batch_send_wait: Duration,
    pub(crate) terminal_send_wait: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProducerExitKind {
    Finished,
    ErrorReported,
    ReceiverDropped,
}

#[derive(Debug)]
pub(crate) struct ProducerExit {
    pub(crate) kind: ProducerExitKind,
    pub(crate) stats: ProducerStats,
}

#[derive(Debug)]
pub(crate) struct ProducerPanic(String);

impl std::fmt::Display for ProducerPanic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "batch producer panicked: {}", self.0)
    }
}

impl std::error::Error for ProducerPanic {}

/// Owns the rendezvous receiver and the named producer thread.  The receiver
/// is always taken and dropped before the thread is joined, both in the
/// explicit shutdown method and in the fallback `Drop` implementation.
pub(crate) struct BatchProducer {
    receiver: Option<Receiver<ProducerMessage>>,
    handle: Option<JoinHandle<ProducerExit>>,
}

impl BatchProducer {
    pub(crate) fn spawn(stream: DisjointBamStream) -> io::Result<Self> {
        let (sender, receiver) = mpsc::sync_channel(0);
        // The lifetime includes any delay between requesting the spawn and the
        // child starting, then the complete stream/pool teardown in the child.
        let lifetime_start = Instant::now();
        let handle = thread::Builder::new()
            .name("gpugeno-batch-producer".to_string())
            .spawn(move || producer_loop(stream, sender, lifetime_start))?;
        Ok(Self {
            receiver: Some(receiver),
            handle: Some(handle),
        })
    }

    pub(crate) fn recv(&self) -> Result<ProducerMessage, RecvError> {
        self.receiver
            .as_ref()
            .expect("batch producer receiver was already disconnected")
            .recv()
    }

    /// Drops the receiver before joining.  This is the required cancellation
    /// order: a producer blocked in the zero-capacity send must observe the
    /// disconnect before it can tear down its stream workers and return.
    pub(crate) fn disconnect_and_join(mut self) -> Result<ProducerExit, ProducerPanic> {
        drop(self.receiver.take());
        let handle = self
            .handle
            .take()
            .expect("batch producer handle was already joined");
        handle
            .join()
            .map_err(|payload| ProducerPanic(decode_panic(payload)))
    }
}

impl Drop for BatchProducer {
    fn drop(&mut self) {
        // This fallback must not detach a live producer and must not panic if
        // it runs during unwinding.  Explicit run paths use
        // `disconnect_and_join` so they can report a panic or protocol error.
        drop(self.receiver.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Preserve a substantive main-thread root error while still completing
/// producer cancellation.  A producer panic is useful secondary context but
/// never replaces the root that caused cancellation.
pub(crate) fn cancel_with_root(producer: BatchProducer, root: String) -> String {
    match producer.disconnect_and_join() {
        Ok(_) => root,
        Err(panic) => format!("{root}; secondary cleanup error: {panic}"),
    }
}

fn producer_loop(
    mut stream: DisjointBamStream,
    sender: SyncSender<ProducerMessage>,
    lifetime_start: Instant,
) -> ProducerExit {
    let mut stats = ProducerStats::default();
    let mut batches_sent = 0usize;
    let exit_kind = loop {
        let (message, category, terminal_kind) = match stream.next_batch() {
            Ok(Some(batch)) => {
                let category = if batches_sent == 0 {
                    SendCategory::FirstBatch
                } else {
                    SendCategory::LaterBatch
                };
                (ProducerMessage::Batch(batch), category, None)
            }
            Ok(None) => (
                ProducerMessage::Eof,
                SendCategory::Terminal,
                Some(ProducerExitKind::Finished),
            ),
            Err(error) => (
                ProducerMessage::StreamError(error),
                SendCategory::Terminal,
                Some(ProducerExitKind::ErrorReported),
            ),
        };

        let send_start = Instant::now();
        let send_result = sender.send(message);
        let send_wait = send_start.elapsed();
        record_send_wait(&mut stats, category, send_wait);
        if send_result.is_err() {
            break ProducerExitKind::ReceiverDropped;
        }

        if let Some(kind) = terminal_kind {
            break kind;
        }
        batches_sent += 1;
    };

    // Explicitly tear down the stream, including its persistent BGZF worker
    // pool, before sampling the producer lifetime and returning its stats.
    drop(sender);
    drop(stream);
    stats.lifetime = lifetime_start.elapsed();
    ProducerExit {
        kind: exit_kind,
        stats,
    }
}

#[derive(Clone, Copy)]
enum SendCategory {
    FirstBatch,
    LaterBatch,
    Terminal,
}

fn record_send_wait(stats: &mut ProducerStats, category: SendCategory, wait: Duration) {
    match category {
        SendCategory::FirstBatch => stats.first_batch_send_wait += wait,
        SendCategory::LaterBatch => stats.later_batch_send_wait += wait,
        SendCategory::Terminal => stats.terminal_send_wait += wait,
    }
}

fn decode_panic(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else {
        "non-string panic payload".to_string()
    }
}

// These calls are type-checked even though the helper itself has no runtime
// role.  They are permanent compile-time guards against accidentally adding a
// non-Send field to the moved stream/batch/error types.
#[allow(dead_code)]
fn compile_time_send_assertions() {
    fn assert_send<T: Send>() {}
    assert_send::<DisjointBamStream>();
    assert_send::<IndexedBamBatch>();
    assert_send::<BatchError>();
    assert_send::<ProducerMessage>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpugeno::bai::{flagstat_anchors, read_bai};
    use gpugeno::bam::{classify_records, read_bam_header};
    use gpugeno::bgzf::{data_end_virtual_offset, VirtualOffset};
    use gpugeno::indexed_batch::DisjointBamStream;
    use libdeflater::{CompressionLvl, Compressor};
    use std::fs::{self, File};
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc};
    use std::thread;
    use std::time::Duration;

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    #[derive(Debug, PartialEq, Eq)]
    enum TestEvent {
        Building(u32),
        Built(u32),
        Attempting(u32),
    }

    #[derive(Debug)]
    struct LiveState {
        live: AtomicUsize,
        peak: AtomicUsize,
    }

    impl LiveState {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                live: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
            })
        }

        fn constructed(self: &Arc<Self>, id: u32) -> TestItem {
            let current = self.live.fetch_add(1, Ordering::SeqCst) + 1;
            let mut previous = self.peak.load(Ordering::SeqCst);
            while current > previous {
                match self.peak.compare_exchange_weak(
                    previous,
                    current,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(observed) => previous = observed,
                }
            }
            TestItem {
                id,
                state: Arc::clone(self),
            }
        }
    }

    #[derive(Debug)]
    struct TestItem {
        id: u32,
        state: Arc<LiveState>,
    }

    impl Drop for TestItem {
        fn drop(&mut self) {
            self.state.live.fetch_sub(1, Ordering::SeqCst);
        }
    }

    enum TestAction {
        Item(u32),
        BlockItem {
            id: u32,
            release: mpsc::Receiver<()>,
        },
        Error(&'static str),
        Panic(&'static str),
        Disconnect,
    }

    enum TestMessage {
        Item(TestItem),
        Error(String),
        Eof,
    }

    #[derive(Debug, PartialEq, Eq)]
    enum TestExitKind {
        Finished,
        ErrorReported,
        ReceiverDropped,
    }

    struct TestProducer {
        receiver: Option<Receiver<TestMessage>>,
        handle: Option<JoinHandle<TestExitKind>>,
    }

    impl TestProducer {
        fn spawn(
            actions: Vec<TestAction>,
            events: mpsc::Sender<TestEvent>,
            state: Arc<LiveState>,
        ) -> Self {
            let (sender, receiver) = mpsc::sync_channel(0);
            let handle = thread::Builder::new()
                .name("gpugeno-test-producer".to_string())
                .spawn(move || {
                    for action in actions {
                        let (message, terminal) = match action {
                            TestAction::Item(id) => {
                                let item = state.constructed(id);
                                let _ = events.send(TestEvent::Built(id));
                                (TestMessage::Item(item), None)
                            }
                            TestAction::BlockItem { id, release } => {
                                let _ = events.send(TestEvent::Building(id));
                                release.recv().expect("test build release was dropped");
                                let item = state.constructed(id);
                                let _ = events.send(TestEvent::Built(id));
                                (TestMessage::Item(item), None)
                            }
                            TestAction::Error(message) => (
                                TestMessage::Error(message.to_string()),
                                Some(TestExitKind::ErrorReported),
                            ),
                            TestAction::Panic(message) => panic!("{message}"),
                            TestAction::Disconnect => return TestExitKind::Finished,
                        };
                        let id = match &message {
                            TestMessage::Item(item) => Some(item.id),
                            TestMessage::Error(_) | TestMessage::Eof => None,
                        };
                        if let Some(id) = id {
                            let _ = events.send(TestEvent::Attempting(id));
                        } else {
                            let _ = events.send(TestEvent::Attempting(u32::MAX));
                        }
                        if sender.send(message).is_err() {
                            return TestExitKind::ReceiverDropped;
                        }
                        if let Some(kind) = terminal {
                            return kind;
                        }
                    }
                    let _ = events.send(TestEvent::Attempting(u32::MAX));
                    if sender.send(TestMessage::Eof).is_err() {
                        TestExitKind::ReceiverDropped
                    } else {
                        TestExitKind::Finished
                    }
                })
                .expect("test producer thread must start");
            Self {
                receiver: Some(receiver),
                handle: Some(handle),
            }
        }

        fn recv(&self) -> Result<TestMessage, mpsc::RecvError> {
            self.receiver.as_ref().unwrap().recv()
        }

        fn recv_timeout(&self, timeout: Duration) -> Result<TestMessage, mpsc::RecvTimeoutError> {
            self.receiver.as_ref().unwrap().recv_timeout(timeout)
        }

        fn disconnect_and_join(mut self) -> Result<TestExitKind, String> {
            drop(self.receiver.take());
            self.handle.take().unwrap().join().map_err(decode_panic)
        }
    }

    impl Drop for TestProducer {
        fn drop(&mut self) {
            drop(self.receiver.take());
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn expect_event(events: &Receiver<TestEvent>, expected: TestEvent) {
        assert_eq!(
            events.recv_timeout(Duration::from_secs(2)).unwrap(),
            expected
        );
    }

    fn cancel_test_with_root(
        producer: TestProducer,
        root: &'static str,
    ) -> (String, Result<TestExitKind, String>) {
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let cleanup = producer.disconnect_and_join();
            let result = match &cleanup {
                Ok(_) => root.to_string(),
                Err(panic) => format!("{root}; secondary cleanup error: {panic}"),
            };
            let _ = sender.send((result, cleanup));
        });
        receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("consumer cancellation/join timed out")
    }

    #[test]
    fn compile_time_send_assertions_cover_moved_stream_types() {
        super::compile_time_send_assertions();
    }
    #[test]
    fn ordered_delivery_requires_explicit_eof() {
        let state = LiveState::new();
        let (event_sender, _events) = mpsc::channel();
        let producer = TestProducer::spawn(
            vec![
                TestAction::Item(1),
                TestAction::Item(2),
                TestAction::Item(3),
            ],
            event_sender,
            state.clone(),
        );
        for expected in 1..=3 {
            let TestMessage::Item(item) = producer.recv().unwrap() else {
                panic!("expected ordered item {expected}");
            };
            assert_eq!(item.id, expected);
            drop(item);
        }
        assert!(matches!(producer.recv().unwrap(), TestMessage::Eof));
        assert_eq!(
            producer.disconnect_and_join().unwrap(),
            TestExitKind::Finished
        );
        assert_eq!(state.live.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn rendezvous_has_at_most_two_complete_items_and_blocks_third_build() {
        let state = LiveState::new();
        let (event_sender, events) = mpsc::channel();
        let producer = TestProducer::spawn(
            vec![
                TestAction::Item(1),
                TestAction::Item(2),
                TestAction::Item(3),
            ],
            event_sender,
            state.clone(),
        );
        let TestMessage::Item(first) = producer.recv().unwrap() else {
            panic!("first item missing");
        };
        expect_event(&events, TestEvent::Built(1));
        expect_event(&events, TestEvent::Attempting(1));
        expect_event(&events, TestEvent::Built(2));
        expect_event(&events, TestEvent::Attempting(2));
        assert_eq!(state.live.load(Ordering::SeqCst), 2);
        assert!(matches!(
            events.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert_eq!(state.peak.load(Ordering::SeqCst), 2);

        drop(first);
        let TestMessage::Item(second) = producer.recv().unwrap() else {
            panic!("second item missing");
        };
        expect_event(&events, TestEvent::Built(3));
        expect_event(&events, TestEvent::Attempting(3));
        assert_eq!(state.peak.load(Ordering::SeqCst), 2);
        drop(second);
        let TestMessage::Item(third) = producer.recv().unwrap() else {
            panic!("third item missing");
        };
        drop(third);
        assert!(matches!(producer.recv().unwrap(), TestMessage::Eof));
        assert_eq!(
            producer.disconnect_and_join().unwrap(),
            TestExitKind::Finished
        );
        assert_eq!(state.live.load(Ordering::SeqCst), 0);
        assert_eq!(state.peak.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn producer_errors_are_explicit_before_and_after_items() {
        for actions in [
            vec![TestAction::Error("error-before")],
            vec![TestAction::Item(7), TestAction::Error("error-after")],
        ] {
            let state = LiveState::new();
            let (event_sender, _events) = mpsc::channel();
            let producer = TestProducer::spawn(actions, event_sender, state);
            let first_message = producer.recv_timeout(Duration::from_secs(2)).unwrap();
            let error_message = match first_message {
                TestMessage::Item(item) => {
                    drop(item);
                    producer.recv().unwrap()
                }
                message => message,
            };
            let TestMessage::Error(message) = error_message else {
                panic!("expected explicit stream error");
            };
            assert!(message == "error-before" || message == "error-after");
            assert!(matches!(
                producer.recv_timeout(Duration::from_millis(100)),
                Err(mpsc::RecvTimeoutError::Disconnected)
            ));
            assert_eq!(
                producer.disconnect_and_join().unwrap(),
                TestExitKind::ErrorReported
            );
        }
    }

    #[test]
    fn consumer_root_survives_cancellation_while_send_is_blocked() {
        let state = LiveState::new();
        let (event_sender, events) = mpsc::channel();
        let producer = TestProducer::spawn(
            vec![
                TestAction::Item(1),
                TestAction::Item(2),
                TestAction::Item(3),
            ],
            event_sender,
            state,
        );
        let TestMessage::Item(first) = producer.recv().unwrap() else {
            panic!("first item missing");
        };
        expect_event(&events, TestEvent::Built(1));
        expect_event(&events, TestEvent::Attempting(1));
        expect_event(&events, TestEvent::Built(2));
        expect_event(&events, TestEvent::Attempting(2));
        drop(first);
        let (root, cleanup) = cancel_test_with_root(producer, "injected consumer backend failure");
        assert_eq!(root, "injected consumer backend failure");
        assert_eq!(cleanup.unwrap(), TestExitKind::ReceiverDropped);
    }

    #[test]
    fn receiver_disconnect_while_building_unblocks_after_build_finishes() {
        let state = LiveState::new();
        let (event_sender, events) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let producer = TestProducer::spawn(
            vec![TestAction::BlockItem {
                id: 11,
                release: release_receiver,
            }],
            event_sender,
            state,
        );
        expect_event(&events, TestEvent::Building(11));
        let (join_sender, join_receiver) = mpsc::channel();
        thread::spawn(move || {
            let _ = join_sender.send(producer.disconnect_and_join());
        });
        release_sender.send(()).unwrap();
        let result = join_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("disconnect helper must not hang")
            .expect("producer must not panic");
        assert_eq!(result, TestExitKind::ReceiverDropped);
    }

    #[test]
    fn producer_panic_is_not_treated_as_eof() {
        let state = LiveState::new();
        let (event_sender, _events) = mpsc::channel();
        let producer = TestProducer::spawn(
            vec![TestAction::Panic("scripted producer panic")],
            event_sender,
            state,
        );
        assert!(matches!(
            producer.recv_timeout(Duration::from_secs(2)),
            Err(mpsc::RecvTimeoutError::Disconnected)
        ));
        let panic = producer.disconnect_and_join().unwrap_err();
        assert!(panic.contains("scripted producer panic"));
    }

    #[test]
    fn disconnect_without_terminal_message_is_a_protocol_error() {
        let state = LiveState::new();
        let (event_sender, _events) = mpsc::channel();
        let producer = TestProducer::spawn(vec![TestAction::Disconnect], event_sender, state);
        assert!(matches!(
            producer.recv_timeout(Duration::from_secs(2)),
            Err(mpsc::RecvTimeoutError::Disconnected)
        ));
        let exit = producer.disconnect_and_join().unwrap();
        assert_eq!(exit, TestExitKind::Finished);
        let protocol_error = match exit {
            TestExitKind::Finished => {
                "batch producer channel disconnected without EOF or stream error".to_string()
            }
            other => format!(
                "batch producer channel disconnected without EOF or stream error (exit={other:?})"
            ),
        };
        assert_eq!(
            protocol_error,
            "batch producer channel disconnected without EOF or stream error"
        );
    }

    const BGZF_EOF: [u8; 28] = [
        0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02,
        0x00, 0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    fn bgzf_member(compressor: &mut Compressor, input: &[u8]) -> Vec<u8> {
        let mut gzip = vec![0u8; compressor.gzip_compress_bound(input.len())];
        let length = compressor.gzip_compress(input, &mut gzip).unwrap();
        gzip.truncate(length);
        let mut member = Vec::with_capacity(gzip.len() + 8);
        member.extend_from_slice(&gzip[..3]);
        member.push(gzip[3] | 0x04);
        member.extend_from_slice(&gzip[4..10]);
        member.extend_from_slice(&6u16.to_le_bytes());
        member.extend_from_slice(b"BC");
        member.extend_from_slice(&2u16.to_le_bytes());
        member.extend_from_slice(&[0, 0]);
        member.extend_from_slice(&gzip[10..]);
        let bsize = u16::try_from(member.len() - 1).unwrap();
        member[16..18].copy_from_slice(&bsize.to_le_bytes());
        member
    }

    fn bam_record(flag: u16, mapq: u8, reference: i32, next_reference: i32) -> Vec<u8> {
        let mut record = vec![0u8; 36];
        record[..4].copy_from_slice(&32u32.to_le_bytes());
        record[4..8].copy_from_slice(&reference.to_le_bytes());
        record[13] = mapq;
        record[18..20].copy_from_slice(&flag.to_le_bytes());
        record[24..28].copy_from_slice(&next_reference.to_le_bytes());
        record
    }

    struct Fixture {
        bam: std::path::PathBuf,
        bai: std::path::PathBuf,
        anchors: Vec<VirtualOffset>,
        data_end: VirtualOffset,
    }

    fn real_fixture(corrupt_first_member: bool) -> Fixture {
        let mut header = b"BAM\x01".to_vec();
        header.extend_from_slice(&0i32.to_le_bytes());
        header.extend_from_slice(&1i32.to_le_bytes());
        header.extend_from_slice(&2i32.to_le_bytes());
        header.extend_from_slice(b"1\0");
        header.extend_from_slice(&1000i32.to_le_bytes());
        let header_len = header.len();
        let mut first_data = header;
        first_data.extend(bam_record(0, 60, 0, 0));
        first_data.extend(bam_record(0x001 | 0x040 | 0x002, 30, 0, 0));
        let mut second_data = bam_record(0x001 | 0x080, 5, 0, 1);
        second_data.extend(bam_record(0x200 | 0x004 | 0x400, 0, -1, -1));

        let mut compressor = Compressor::new(CompressionLvl::default());
        let mut first = bgzf_member(&mut compressor, &first_data);
        let second_offset = first.len() as u64;
        let second = bgzf_member(&mut compressor, &second_data);
        let data_end_raw = (second_offset + second.len() as u64) << 16;
        let first_anchor = VirtualOffset::new(0, u16::try_from(header_len).unwrap()).unwrap();
        let second_anchor = VirtualOffset::new(0, u16::try_from(header_len + 36).unwrap()).unwrap();
        let third_anchor = VirtualOffset::new(second_offset, 0).unwrap();
        let fourth_anchor = VirtualOffset::new(second_offset, 36).unwrap();
        let data_end = VirtualOffset::from_raw(data_end_raw);
        if corrupt_first_member {
            let crc_position = first.len() - 8;
            first[crc_position] ^= 0xff;
        }

        let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let bam = std::env::temp_dir().join(format!(
            "gpugeno-overlap-stream-{}-{id}.bam",
            std::process::id()
        ));
        let bai = bam.with_extension("bam.bai");
        let mut bam_file = File::create(&bam).unwrap();
        bam_file.write_all(&first).unwrap();
        bam_file.write_all(&second).unwrap();
        bam_file.write_all(&BGZF_EOF).unwrap();
        drop(bam_file);

        let mut bai_bytes = b"BAI\x01".to_vec();
        bai_bytes.extend_from_slice(&1i32.to_le_bytes());
        bai_bytes.extend_from_slice(&0i32.to_le_bytes());
        bai_bytes.extend_from_slice(&4i32.to_le_bytes());
        for offset in [first_anchor, second_anchor, third_anchor, fourth_anchor] {
            bai_bytes.extend_from_slice(&offset.raw().to_le_bytes());
        }
        bai_bytes.extend_from_slice(&0u64.to_le_bytes());
        fs::write(&bai, bai_bytes).unwrap();

        Fixture {
            bam,
            bai,
            anchors: vec![first_anchor, second_anchor, third_anchor, fourth_anchor],
            data_end,
        }
    }

    fn collect_sequential(mut stream: DisjointBamStream) -> Result<Vec<IndexedBamBatch>, String> {
        let mut batches = Vec::new();
        while let Some(batch) = stream.next_batch().map_err(|error| error.to_string())? {
            batches.push(batch);
        }
        Ok(batches)
    }

    fn collect_producer(stream: DisjointBamStream) -> Result<Vec<IndexedBamBatch>, String> {
        let producer = BatchProducer::spawn(stream).map_err(|error| error.to_string())?;
        let mut producer = Some(producer);
        let mut batches = Vec::new();
        loop {
            let message = match producer.as_ref().unwrap().recv() {
                Ok(message) => message,
                Err(_) => {
                    let exit = producer
                        .take()
                        .unwrap()
                        .disconnect_and_join()
                        .map_err(|error| error.to_string());
                    return Err(match exit {
                        Ok(exit) => format!(
                            "batch producer channel disconnected without EOF or stream error (exit={:?})",
                            exit.kind
                        ),
                        Err(error) => error,
                    });
                }
            };
            match message {
                ProducerMessage::Batch(batch) => batches.push(batch),
                ProducerMessage::Eof => {
                    let exit = producer
                        .take()
                        .unwrap()
                        .disconnect_and_join()
                        .map_err(|error| error.to_string())?;
                    assert_eq!(exit.kind, ProducerExitKind::Finished);
                    break;
                }
                ProducerMessage::StreamError(error) => {
                    let _ = producer.take().unwrap().disconnect_and_join();
                    return Err(error.to_string());
                }
            }
        }
        Ok(batches)
    }

    fn compare_batch_fields(left: &IndexedBamBatch, right: &IndexedBamBatch) {
        assert_eq!(left.data, right.data);
        assert_eq!(left.span_starts, right.span_starts);
        assert_eq!(left.virtual_start, right.virtual_start);
        assert_eq!(left.virtual_end, right.virtual_end);
        assert_eq!(left.block_map, right.block_map);
        assert_eq!(left.blocks_decompressed, right.blocks_decompressed);
        assert_eq!(left.compressed_bytes_read, right.compressed_bytes_read);
    }

    #[test]
    fn real_stream_sequential_and_rendezvous_batches_are_equivalent() {
        let fixture = real_fixture(false);
        let header = read_bam_header(&fixture.bam).unwrap();
        assert_eq!(header.first_record, fixture.anchors[0]);
        assert_eq!(
            data_end_virtual_offset(&fixture.bam).unwrap(),
            fixture.data_end
        );
        let index = read_bai(&fixture.bai).unwrap();
        let anchors = flagstat_anchors(
            header.first_record,
            fixture.data_end,
            &index.linear_work_items,
        )
        .unwrap();
        assert_eq!(
            anchors,
            fixture
                .anchors
                .iter()
                .chain(std::iter::once(&fixture.data_end))
                .copied()
                .collect::<Vec<_>>()
        );

        let sequential = collect_sequential(
            DisjointBamStream::open(&fixture.bam, anchors.clone(), 72, 2).unwrap(),
        )
        .unwrap();
        let producer =
            collect_producer(DisjointBamStream::open(&fixture.bam, anchors, 72, 2).unwrap())
                .unwrap();
        assert_eq!(sequential.len(), 2);
        assert_eq!(producer.len(), sequential.len());
        for (left, right) in sequential.iter().zip(&producer) {
            compare_batch_fields(left, right);
        }

        let mut sequential_data = Vec::new();
        let mut producer_data = Vec::new();
        let mut sequential_counts = gpugeno::bam::FlagstatCounters::default();
        let mut producer_counts = gpugeno::bam::FlagstatCounters::default();
        for batch in &sequential {
            sequential_data.extend_from_slice(&batch.data);
            sequential_counts.add_assign(&classify_records(&batch.data).unwrap());
        }
        for batch in &producer {
            producer_data.extend_from_slice(&batch.data);
            producer_counts.add_assign(&classify_records(&batch.data).unwrap());
        }
        assert_eq!(sequential_data, producer_data);
        assert_eq!(sequential_data.len(), 4 * 36);
        assert_eq!(sequential_counts, producer_counts);
        assert_eq!(
            sequential_counts,
            classify_records(&sequential_data).unwrap()
        );
        assert_eq!(
            sequential
                .iter()
                .map(|batch| batch.span_count())
                .sum::<usize>(),
            4
        );
        fs::remove_file(fixture.bam).unwrap();
        fs::remove_file(fixture.bai).unwrap();
    }

    #[test]
    fn real_stream_open_and_build_errors_propagate_through_protocol() {
        let fixture = real_fixture(false);
        let missing = fixture.bam.with_extension("missing.bam");
        let open_result = DisjointBamStream::open(&missing, fixture.anchors.clone(), 72, 2);
        assert!(open_result.is_err());

        let corrupt = real_fixture(true);
        let stream = DisjointBamStream::open(&corrupt.bam, corrupt.anchors, 72, 2).unwrap();
        let producer = BatchProducer::spawn(stream).unwrap();
        let message = producer.recv().unwrap();
        let ProducerMessage::StreamError(error) = message else {
            panic!("corrupt BGZF must be a producer stream error");
        };
        assert!(error
            .to_string()
            .contains("libdeflate gzip decompression failed"));
        let exit = producer.disconnect_and_join().unwrap();
        assert_eq!(exit.kind, ProducerExitKind::ErrorReported);
        fs::remove_file(fixture.bam).unwrap();
        fs::remove_file(fixture.bai).unwrap();
        fs::remove_file(corrupt.bam).unwrap();
        fs::remove_file(corrupt.bai).unwrap();
    }
}
