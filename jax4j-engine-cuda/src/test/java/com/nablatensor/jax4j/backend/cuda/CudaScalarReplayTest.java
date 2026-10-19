package com.nablatensor.jax4j.backend.cuda;

import com.nablatensor.jax4j.engine.*;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "jax4j.cuda.tests", matches = "true")
class CudaScalarReplayTest {
  private Stream<AadEngine> engines() {
    return Stream.of(new CudaAadEngine(), new CudaInterpEngine(), new CudaEagerEngine());
  }

  @TestFactory Stream<DynamicTest> randomReplayMatchesCpu() {
    return engines().flatMap(engine -> Stream.of(AadOptions.Precision.values()).map(precision ->
        DynamicTest.dynamicTest(engine.name() + " random " + precision, () -> {
          assertTrue(engine.isAvailable(), "CUDA explicitly required");
          var tape = AadRecorder.record(rec -> {
            var x = rec.input("x", 2);
            var z = rec.randn();
            rec.output(x.mul(z).add(z.mul(z)));
          });
          var options = new AadOptions(precision, true).withThreads(1);
          try (var gpu = engine.compile(tape, options);
               var cpu = new com.nablatensor.jax4j.engine.cpu.CpuAadEngine().compile(
                   tape, options.withPrecision(AadOptions.Precision.FLOAT64))) {
            for (long offset : new long[] {0, 1031}) {
              var expected = cpu.replay(1025, offset, 42);
              var actual = gpu.replay(1025, offset, 42);
              assertEquals(expected.value(), actual.value(), 2e-5);
              assertEquals(expected.gradient("x"), actual.gradient("x"), 2e-5);
            }
          }
        })));
  }

  @TestFactory Stream<DynamicTest> valuesGradientsAndInputUpdates() {
    return engines().flatMap(engine -> Stream.of(AadOptions.Precision.values()).flatMap(precision ->
        Stream.of(true, false).map(adjoints -> DynamicTest.dynamicTest(
            engine.name() + " " + precision + " adjoints=" + adjoints, () -> {
              assertTrue(engine.isAvailable(), "CUDA explicitly required");
              var tape = AadRecorder.record(rec -> {
                var x = rec.input("x", 2);
                var y = rec.input("y", 3);
                rec.output(x.mul(x).mul(y).add(x.exp()).add(y.exp()));
              });
              var options = new AadOptions(precision, adjoints);
              double tolerance = precision == AadOptions.Precision.FLOAT32 ? 2e-4 : 1e-10;
              try (var executable = engine.compile(tape, options)) {
                assertEquals(engine.name(), executable.engineName());
                for (double x : new double[] {2, -0.5}) {
                  executable.setInput("x", x);
                  var result = executable.replay(1025, 17, 42);
                  assertEquals(x * x * 3 + Math.exp(x) + Math.exp(3), result.value(), tolerance);
                  if (adjoints) {
                    assertEquals(6 * x + Math.exp(x), result.gradient("x"), tolerance);
                    assertEquals(x * x + Math.exp(3), result.gradient("y"), tolerance);
                  }
                }
              }
            }))));
  }
}
