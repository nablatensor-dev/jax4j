package com.nablatensor.jax4j.backend.vulkan;

import java.util.concurrent.atomic.AtomicLong;

/** Counters include native Vulkan work only; snapshots do not synchronize the queue. */
public final class VulkanDiagnostics {
  private static final AtomicLong dispatches=new AtomicLong(), allocations=new AtomicLong(),
      allocatedBytes=new AtomicLong(), uploads=new AtomicLong(), downloads=new AtomicLong(), compilations=new AtomicLong();
  private static final java.util.Map<Long,Long> sizes=new java.util.HashMap<>();
  private static long liveBytes,peakBytes;
  private VulkanDiagnostics() {}
  public record Snapshot(long dispatches,long allocations,long allocatedBytes,long hostToDeviceBytes,
                         long deviceToHostBytes,long compilations,long liveDeviceBytes,long peakDeviceBytes) {
    public Snapshot minus(Snapshot before) {
      return new Snapshot(dispatches-before.dispatches,allocations-before.allocations,
          allocatedBytes-before.allocatedBytes,hostToDeviceBytes-before.hostToDeviceBytes,
          deviceToHostBytes-before.deviceToHostBytes,compilations-before.compilations,liveDeviceBytes-before.liveDeviceBytes,peakDeviceBytes);
    }
  }
  public static synchronized Snapshot snapshot() { return new Snapshot(dispatches.get(),allocations.get(),allocatedBytes.get(),uploads.get(),downloads.get(),compilations.get(),liveBytes,peakBytes); }
  static synchronized void allocated(long buffer,long bytes) { allocations.incrementAndGet(); allocatedBytes.addAndGet(bytes); sizes.put(buffer,bytes); liveBytes+=bytes; peakBytes=Math.max(peakBytes,liveBytes); }
  static synchronized void freed(long buffer) { Long bytes=sizes.remove(buffer); if(bytes!=null)liveBytes-=bytes; }
  static void uploaded(long bytes) { uploads.addAndGet(bytes); }
  static void downloaded(long bytes) { downloads.addAndGet(bytes); }
  static void dispatched() { dispatches.incrementAndGet(); }
  static void compiled() { compilations.incrementAndGet(); }
}
