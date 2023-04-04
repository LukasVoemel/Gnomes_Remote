# typed: true

# DO NOT EDIT MANUALLY
# This is an autogenerated file for types exported from the `parallel` gem.
# Please instead update this file by running `bin/tapioca gem parallel`.

module Parallel
  extend ::Parallel::ProcessorCount

  class << self
    def all?(*args, &block); end
    def any?(*args, &block); end
    def each(array, options = T.unsafe(nil), &block); end
    def each_with_index(array, options = T.unsafe(nil), &block); end
    def flat_map(*args, &block); end
    def in_processes(options = T.unsafe(nil), &block); end
    def in_threads(options = T.unsafe(nil)); end
    def map(source, options = T.unsafe(nil), &block); end
    def map_with_index(array, options = T.unsafe(nil), &block); end
    def worker_number; end
    def worker_number=(worker_num); end

    private

    def add_progress_bar!(job_factory, options); end
    def call_with_index(item, index, options, &block); end
    def create_workers(job_factory, options, &block); end
    def extract_count_from_options(options); end
    def instrument_finish(item, index, result, options); end
    def instrument_start(item, index, options); end
    def process_incoming_jobs(read, write, job_factory, options, &block); end
    def replace_worker(job_factory, workers, index, options, blk); end
    def with_instrumentation(item, index, options); end
    def work_direct(job_factory, options, &block); end
    def work_in_processes(job_factory, options, &blk); end
    def work_in_ractors(job_factory, options); end
    def work_in_threads(job_factory, options, &block); end
    def worker(job_factory, options, &block); end
  end
end

class Parallel::Break < ::StandardError
  def initialize(value = T.unsafe(nil)); end

  def value; end
end

class Parallel::DeadWorker < ::StandardError; end

class Parallel::ExceptionWrapper
  def initialize(exception); end

  def exception; end
end

class Parallel::JobFactory
  def initialize(source, mutex); end

  def next; end
  def pack(item, index); end
  def size; end
  def unpack(data); end

  private

  def producer?; end
  def queue_wrapper(array); end
end

class Parallel::Kill < ::Parallel::Break; end

module Parallel::ProcessorCount
  def physical_processor_count; end
  def processor_count; end
end

Parallel::Stop = T.let(T.unsafe(nil), Object)

class Parallel::UndumpableException < ::StandardError
  def initialize(original); end

  def backtrace; end
end

class Parallel::UserInterruptHandler
  class << self
    def kill(thing); end
    def kill_on_ctrl_c(pids, options); end

    private

    def restore_interrupt(old, signal); end
    def trap_interrupt(signal); end
  end
end

Parallel::UserInterruptHandler::INTERRUPT_SIGNAL = T.let(T.unsafe(nil), Symbol)
Parallel::VERSION = T.let(T.unsafe(nil), String)
Parallel::Version = T.let(T.unsafe(nil), String)

class Parallel::Worker
  def initialize(read, write, pid); end

  def close_pipes; end
  def pid; end
  def read; end
  def stop; end
  def thread; end
  def thread=(_arg0); end
  def work(data); end
  def write; end

  private

  def wait; end
end
