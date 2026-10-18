package com.jax4j.backend.wgpu;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class WgpuAvailabilityTest {
  private Process probe(String... args) throws Exception {
    var command = new java.util.ArrayList<String>();
    command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
    command.add("--enable-native-access=ALL-UNNAMED");
    command.add("-Djax4j.wgpu.native_path=/nonexistent/jax4j-test/libwgpu_native.so");
    command.add("-cp");
    command.add(System.getProperty("java.class.path"));
    command.add(WgpuInfo.class.getName());
    java.util.Collections.addAll(command, args);
    return new ProcessBuilder(command).redirectErrorStream(true).start();
  }

  @Test
  void cpuStartupDoesNotRequireTheNativeLibrary() throws Exception {
    Process child = probe("--cpu");
    assertTrue(child.waitFor(10, TimeUnit.SECONDS));
    String output = new String(child.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
    assertEquals(0, child.exitValue(), output);
    assertTrue(output.contains("CPU OK"), output);
  }

  @Test
  void explicitUnavailableWgpuDoesNotFallBackToCpu() throws Exception {
    Process child = probe();
    assertTrue(child.waitFor(10, TimeUnit.SECONDS));
    String output = new String(child.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
    assertEquals(2, child.exitValue(), output);
    assertTrue(output.contains("Explicit wgpu request rejected"), output);
  }
}
