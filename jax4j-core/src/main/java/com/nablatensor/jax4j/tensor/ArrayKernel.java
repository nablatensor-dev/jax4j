package com.nablatensor.jax4j.tensor;

import java.util.Arrays;

/** Shape-specialized, device-local array kernels. Runtime indices remain device inputs. */
public final class ArrayKernel {
  public enum Kind {
    WHERE,
    CONCAT,
    ONEHOT,
    GATHER,
    SCATTER,
    TAKE,
    SLICE,
    UPDATE,
    PROD,
    TOPK,
    SEGMENT,
    PULL,
    PUSH,
    BATCH
  }

  private final Kind kind;
  private final Shape output;
  private final Shape[] inputs;
  private final int axis;
  private final int[] parameters;
  private final ArrayKernel forward;
  private final int differentiated;

  public ArrayKernel(Kind kind, Shape output, Shape[] inputs, int axis, int[] parameters) {
    this(kind, output, inputs, axis, parameters, null, -1);
  }

  private ArrayKernel(
      Kind kind,
      Shape output,
      Shape[] inputs,
      int axis,
      int[] parameters,
      ArrayKernel forward,
      int differentiated) {
    this.kind = kind;
    this.output = output;
    this.inputs = inputs.clone();
    this.axis = axis;
    this.parameters = parameters.clone();
    this.forward = forward;
    this.differentiated = differentiated;
  }

  public Shape output() {
    return output;
  }

  public int inputCount() {
    return inputs.length;
  }

  public boolean derivative() {
    return kind == Kind.PULL || kind == Kind.PUSH || kind == Kind.BATCH && forward.derivative();
  }

  public ArrayKernel batched(int count) {
    if (count < 0) throw new IllegalArgumentException("batch count");
    Shape[] physical = new Shape[inputs.length];
    for (int a = 0; a < inputs.length; a++) physical[a] = prepend(inputs[a], count);
    return new ArrayKernel(
        Kind.BATCH, prepend(output, count), physical, count, parameters, this, -1);
  }

  private static Shape prepend(Shape shape, int count) {
    int[] dims = new int[shape.rank() + 1];
    dims[0] = count;
    System.arraycopy(shape.dims(), 0, dims, 1, shape.rank());
    return Shape.of(dims);
  }

  public ArrayKernel pull(int input) {
    if (kind == Kind.BATCH) return forward.pull(input).batched(axis);
    Shape[] shapes = Arrays.copyOf(inputs, inputs.length + 1);
    shapes[inputs.length] = output;
    return new ArrayKernel(Kind.PULL, inputs[input], shapes, axis, parameters, this, input);
  }

  public ArrayKernel push() {
    if (kind == Kind.BATCH) return forward.push().batched(axis);
    Shape[] shapes = Arrays.copyOf(inputs, inputs.length * 2);
    System.arraycopy(inputs, 0, shapes, inputs.length, inputs.length);
    return new ArrayKernel(Kind.PUSH, output, shapes, axis, parameters, this, -1);
  }

  private int size(Shape s) {
    return Math.toIntExact(s.size());
  }

  private int inner() {
    int n = 1;
    for (int d = axis + 1; d < inputs[0].rank(); d++) n = Math.multiplyExact(n, inputs[0].dim(d));
    return n;
  }

  private int[] starts(float[][] x, int slot, Shape region) {
    int[] starts = new int[inputs[0].rank()];
    for (int d = 0; d < starts.length; d++) {
      float v = x[slot][d];
      if (!Float.isFinite(v) || v != Math.rint(v)) return null;
      starts[d] = (int) Math.max(0, Math.min(v, inputs[0].dim(d) - region.dim(d)));
    }
    return starts;
  }

  private int selected(int i, float[][] x) {
    if (kind == Kind.WHERE) return x[0][i] != 0 ? 1 : 2;
    if (kind == Kind.CONCAT) {
      int c = i / Math.max(1, outputStride(output, axis)) % Math.max(1, output.dim(axis)),
          offset = 0;
      for (int a = 0; a < inputs.length; a++) {
        offset += inputs[a].dim(axis);
        if (c < offset) return a;
      }
    }
    if (kind == Kind.UPDATE) {
      int[] start = starts(x, 2, inputs[1]);
      if (start == null) return 0;
      for (int d = inputs[0].rank() - 1, n = i; d >= 0; d--) {
        int c = n % inputs[0].dim(d);
        n /= inputs[0].dim(d);
        if (c < start[d] || c >= start[d] + inputs[1].dim(d)) return 0;
      }
      return 1;
    }
    return 0;
  }

  private static boolean better(float a, int ai, float b, int bi) {
    if (Float.isNaN(a)) return !Float.isNaN(b) || ai < bi;
    if (Float.isNaN(b)) return false;
    return a > b || a == b && ai < bi;
  }

  private int index(int i, float[][] x) {
    switch (kind) {
      case WHERE:
        return i;
      case CONCAT:
        {
          int selected = selected(i, x), inner = outputStride(output, axis), offset = 0;
          for (int a = 0; a < selected; a++) offset += inputs[a].dim(axis);
          return (i / (Math.max(1, inner * output.dim(axis))) * inputs[selected].dim(axis)
                      + i / Math.max(1, inner) % Math.max(1, output.dim(axis))
                      - offset)
                  * inner
              + i % Math.max(1, inner);
        }
      case GATHER:
        return parameters[i];
      case TAKE:
        {
          int inner = inner(), count = size(inputs[1]), n = inputs[0].dim(axis);
          int q = (i / inner) % Math.max(1, count);
          float v = x[1][q];
          if (!Float.isFinite(v) || v != Math.rint(v) || v < -n || v >= n) return -1;
          int j = (int) v;
          if (j < 0) j += n;
          return (i / (inner * Math.max(1, count)) * n + j) * inner + i % inner;
        }
      case SLICE:
        {
          int[] start = starts(x, 1, output);
          if (start == null) return -1;
          int j = 0, stride = 1, n = i;
          for (int d = output.rank() - 1; d >= 0; d--) {
            int c = n / outputStride(output, d) % Math.max(1, output.dim(d));
            j += (c + start[d]) * stride;
            stride *= inputs[0].dim(d);
          }
          return j;
        }
      case UPDATE:
        {
          if (selected(i, x) == 0) return i;
          int[] start = starts(x, 2, inputs[1]);
          int j = 0, stride = 1;
          for (int d = inputs[0].rank() - 1; d >= 0; d--) {
            int c = i / outputStride(inputs[0], d) % inputs[0].dim(d);
            j += (c - start[d]) * stride;
            stride *= inputs[1].dim(d);
          }
          return j;
        }
      case TOPK:
        {
          int width = inputs[0].dim(inputs[0].rank() - 1),
              k = parameters[0],
              base = i / Math.max(1, k) * width,
              rank = i % Math.max(1, k);
          for (int a = 0; a < width; a++) {
            int above = 0;
            for (int b = 0; b < width; b++)
              if (better(x[0][base + b], b, x[0][base + a], a)) above++;
            if (above == rank) return base + a;
          }
          return -1;
        }
      default:
        throw new IllegalArgumentException("no selection mapping: " + kind);
    }
  }

  private static int outputStride(Shape shape, int axis) {
    int n = 1;
    for (int d = axis + 1; d < shape.rank(); d++) n *= shape.dim(d);
    return n;
  }

  private float weight(int out, int input, int source, float[][] x) {
    if (kind == Kind.PROD) {
      if (input != 0) return 0;
      int inner = inner(),
          width = inputs[0].dim(axis),
          group = (source / (inner * width)) * inner + source % inner;
      if (group != out) return 0;
      float product = 1;
      for (int k = 0; k < width; k++) {
        int j = (out / inner * width + k) * inner + out % inner;
        if (j != source) product *= x[0][j];
      }
      return product;
    }
    if (kind == Kind.SCATTER) return input == 0 && parameters[source] == out ? 1 : 0;
    if (kind == Kind.SEGMENT) {
      if (input != 0) return 0;
      int inner = size(inputs[0]) / Math.max(1, inputs[0].dim(0));
      return inner > 0 && x[1][source / inner] == out / inner && source % inner == out % inner
          ? 1
          : 0;
    }
    if (kind == Kind.ONEHOT || kind == Kind.TOPK && parameters[1] != 0) return 0;
    return selected(out, x) == input && index(out, x) == source ? 1 : 0;
  }

  public float[] evaluate(float[][] x) {
    float[] result = new float[size(output)];
    if (kind == Kind.BATCH) {
      for (int batch = 0; batch < axis; batch++) {
        float[][] sample = new float[inputs.length][];
        for (int a = 0; a < inputs.length; a++) {
          int n = size(forward.inputs[a]);
          sample[a] = Arrays.copyOfRange(x[a], batch * n, (batch + 1) * n);
        }
        float[] values = forward.evaluate(sample);
        System.arraycopy(values, 0, result, batch * values.length, values.length);
      }
      return result;
    }
    for (int i = 0; i < result.length; i++) {
      if (kind == Kind.PULL) {
        float sum = 0;
        for (int j = 0; j < size(forward.output); j++) {
          float w = forward.weight(j, differentiated, i, x);
          if (w != 0) sum += w * x[forward.inputs.length][j];
        }
        result[i] = sum;
      } else if (kind == Kind.PUSH) {
        float sum = 0;
        for (int a = 0; a < forward.inputs.length; a++)
          for (int j = 0; j < size(forward.inputs[a]); j++) {
            float w = forward.weight(i, a, j, x);
            if (w != 0) sum += w * x[forward.inputs.length + a][j];
          }
        result[i] = sum;
      } else if (kind == Kind.ONEHOT) {
        int classes = output.dim(output.rank() - 1);
        result[i] = x[0][i / classes] == i % classes ? 1 : 0;
      } else if (kind == Kind.PROD) {
        int inner = inner(), width = inputs[0].dim(axis);
        float value = 1;
        for (int k = 0; k < width; k++) value *= x[0][(i / inner * width + k) * inner + i % inner];
        result[i] = value;
      } else if (kind == Kind.SCATTER) {
        float sum = 0;
        for (int j = 0; j < parameters.length; j++) if (parameters[j] == i) sum += x[0][j];
        result[i] = sum;
      } else if (kind == Kind.SEGMENT) {
        int inner = size(inputs[0]) / Math.max(1, inputs[0].dim(0));
        float sum = 0;
        for (int j = 0; j < inputs[0].dim(0); j++)
          if (x[1][j] == i / inner) sum += x[0][j * inner + i % inner];
        result[i] = sum;
      } else {
        int j = index(i, x);
        result[i] =
            kind == Kind.TOPK && parameters[1] != 0
                ? j % inputs[0].dim(inputs[0].rank() - 1)
                : j < 0 ? (kind == Kind.GATHER ? 0 : Float.NaN) : x[selected(i, x)][j];
      }
    }
    return result;
  }

  private String mapping(String at, String tag) {
    StringBuilder s = new StringBuilder("int selected" + tag + "=0,index" + tag + "=" + at + ";");
    switch (kind) {
      case CONCAT -> {
        int inner = outputStride(output, axis), offset = 0;
        s.append(
            "int c"
                + tag
                + "=("
                + at
                + "/"
                + Math.max(1, inner)
                + ")%"
                + Math.max(1, output.dim(axis))
                + ";");
        for (int a = 0; a < inputs.length; a++) {
          s.append(
              "if(c"
                  + tag
                  + ">="
                  + offset
                  + " && c"
                  + tag
                  + "<"
                  + (offset + inputs[a].dim(axis))
                  + "){selected"
                  + tag
                  + "="
                  + a
                  + ";index"
                  + tag
                  + "=("
                  + at
                  + "/"
                  + Math.max(1, inner * output.dim(axis))
                  + "*"
                  + inputs[a].dim(axis)
                  + "+c"
                  + tag
                  + "-"
                  + offset
                  + ")*"
                  + inner
                  + "+"
                  + at
                  + "%"
                  + Math.max(1, inner)
                  + ";}");
          offset += inputs[a].dim(axis);
        }
      }
      case WHERE -> s.append("selected" + tag + "=v0[" + at + "]!=0.0?1:2;");
      case GATHER -> s.append("index" + tag + "=ix[" + at + "];");
      case TAKE -> {
        int in = inner(), count = size(inputs[1]), width = inputs[0].dim(axis);
        String z = "z" + tag;
        s.append(
            "float "
                + z
                + "=v1[("
                + at
                + "/"
                + in
                + ")%"
                + Math.max(1, count)
                + "];index"
                + tag
                + "=-1;");
        s.append(
            "if(!isnan("
                + z
                + ") && !isinf("
                + z
                + ") && floor("
                + z
                + ")=="
                + z
                + " && "
                + z
                + ">=-"
                + width
                + ".0 && "
                + z
                + "<"
                + width
                + ".0){int k=int("
                + z
                + ");if(k<0)k+="
                + width
                + ";index"
                + tag
                + "=("
                + at
                + "/"
                + (in * Math.max(1, count))
                + "*"
                + width
                + "+k)*"
                + in
                + "+"
                + at
                + "%"
                + in
                + ";}");
      }
      case SLICE, UPDATE -> {
        Shape region = kind == Kind.SLICE ? output : inputs[1];
        int slot = kind == Kind.SLICE ? 1 : 2;
        s.append("bool valid" + tag + "=true;int mapped" + tag + "=0;bool inside" + tag + "=true;");
        for (int d = 0; d < inputs[0].rank(); d++) {
          String z = "start" + tag + "_" + d, c = "coord" + tag + "_" + d;
          s.append(
              "float "
                  + z
                  + "=v"
                  + slot
                  + "["
                  + d
                  + "];valid"
                  + tag
                  + "=valid"
                  + tag
                  + " && !isnan("
                  + z
                  + ") && !isinf("
                  + z
                  + ") && floor("
                  + z
                  + ")=="
                  + z
                  + ";");
          s.append(
              "int s"
                  + tag
                  + "_"
                  + d
                  + "=int(clamp("
                  + z
                  + ",0.0,"
                  + (inputs[0].dim(d) - region.dim(d))
                  + ".0));");
          Shape coordinates = kind == Kind.SLICE ? output : inputs[0];
          s.append(
              "int "
                  + c
                  + "=("
                  + at
                  + "/"
                  + Math.max(1, outputStride(coordinates, d))
                  + ")%"
                  + Math.max(1, coordinates.dim(d))
                  + ";");
          if (kind == Kind.SLICE)
            s.append(
                "mapped"
                    + tag
                    + "+=("
                    + c
                    + "+s"
                    + tag
                    + "_"
                    + d
                    + ")*"
                    + outputStride(inputs[0], d)
                    + ";");
          else
            s.append(
                "inside"
                    + tag
                    + "=inside"
                    + tag
                    + " && "
                    + c
                    + ">=s"
                    + tag
                    + "_"
                    + d
                    + " && "
                    + c
                    + "<s"
                    + tag
                    + "_"
                    + d
                    + "+"
                    + region.dim(d)
                    + ";mapped"
                    + tag
                    + "+=("
                    + c
                    + "-s"
                    + tag
                    + "_"
                    + d
                    + ")*"
                    + outputStride(region, d)
                    + ";");
        }
        if (kind == Kind.SLICE) s.append("index" + tag + "=valid" + tag + "?mapped" + tag + ":-1;");
        else
          s.append(
              "if(valid"
                  + tag
                  + " && inside"
                  + tag
                  + "){selected"
                  + tag
                  + "=1;index"
                  + tag
                  + "=mapped"
                  + tag
                  + ";}");
      }
      case TOPK -> {
        int width = inputs[0].dim(inputs[0].rank() - 1), k = parameters[0];
        String base = "base" + tag;
        s.append(
            "int "
                + base
                + "="
                + at
                + "/"
                + Math.max(1, k)
                + "*"
                + width
                + ";index"
                + tag
                + "=-1;");
        s.append(
            "for(int a=0;a<"
                + width
                + ";a++){int above=0;for(int b=0;b<"
                + width
                + ";b++){float av=v0["
                + base
                + "+a],bv=v0["
                + base
                + "+b];if((isnan(bv) && (!isnan(av) || b<a)) || (!isnan(bv) && !isnan(av) && (bv>av"
                + " || (bv==av && b<a))))above++;}if(above=="
                + at
                + "%"
                + Math.max(1, k)
                + ")index"
                + tag
                + "="
                + base
                + "+a;}");
      }
      default -> throw new IllegalArgumentException("mapping " + kind);
    }
    return s.toString();
  }

  private String weightSource(String out, String src, int input, String tag) {
    StringBuilder s = new StringBuilder("float weight" + tag + "=0.0;");
    if (kind == Kind.PROD) {
      if (input != 0) return s.toString();
      int inner = inner(), width = inputs[0].dim(axis);
      s.append(
          "if(("
              + src
              + "/"
              + Math.max(1, inner * width)
              + "*"
              + inner
              + "+"
              + src
              + "%"
              + Math.max(1, inner)
              + ")=="
              + out
              + "){weight"
              + tag
              + "=1.0;for(int k=0;k<"
              + width
              + ";k++){int q=("
              + out
              + "/"
              + Math.max(1, inner)
              + "*"
              + width
              + "+k)*"
              + inner
              + "+"
              + out
              + "%"
              + Math.max(1, inner)
              + ";if(q!="
              + src
              + ")weight"
              + tag
              + "*=v0[q];}}");
    } else if (kind == Kind.SCATTER) {
      if (input == 0) s.append("weight" + tag + "=ix[" + src + "]==" + out + "?1.0:0.0;");
    } else if (kind == Kind.SEGMENT) {
      if (input == 0) {
        int inner = size(inputs[0]) / Math.max(1, inputs[0].dim(0));
        s.append(
            "weight"
                + tag
                + "=(v1["
                + src
                + "/"
                + Math.max(1, inner)
                + "]==float("
                + out
                + "/"
                + Math.max(1, inner)
                + ") && "
                + src
                + "%"
                + Math.max(1, inner)
                + "=="
                + out
                + "%"
                + Math.max(1, inner)
                + ")?1.0:0.0;");
      }
    } else if (kind != Kind.ONEHOT && !(kind == Kind.TOPK && parameters[1] != 0))
      s.append(mapping(out, tag))
          .append(
              "weight"
                  + tag
                  + "=(selected"
                  + tag
                  + "=="
                  + input
                  + " && index"
                  + tag
                  + "=="
                  + src
                  + ")?1.0:0.0;");
    return s.toString();
  }

  private String body() {
    if (kind == Kind.BATCH) {
      int depth = 1;
      for (ArrayKernel k = forward; k != null; k = k.forward) depth++;
      String sample = "sample" + depth, batch = "batch" + depth;
      String code =
          forward
              .body()
              .replaceAll("\\bi\\b", sample)
              .replace("outValues[" + sample + "]", "outValues[i]");
      for (int a = 0; a < inputs.length; a++)
        code =
            code.replace(
                "v" + a + "[", "v" + a + "[" + batch + "*" + size(forward.inputs[a]) + "+");
      return "int "
          + sample
          + "=i%"
          + Math.max(1, size(forward.output))
          + ","
          + batch
          + "=i/"
          + Math.max(1, size(forward.output))
          + ";"
          + code;
    }
    StringBuilder s = new StringBuilder("float value=0.0;");
    if (kind == Kind.PULL) {
      s.append("for(int j=0;j<" + size(forward.output) + ";j++){")
          .append(forward.weightSource("j", "i", differentiated, "g"))
          .append("if(weightg!=0.0)value+=weightg*v" + forward.inputs.length + "[j];}");
    } else if (kind == Kind.PUSH) {
      for (int a = 0; a < forward.inputs.length; a++)
        s.append("for(int j=0;j<" + size(forward.inputs[a]) + ";j++){")
            .append(forward.weightSource("i", "j", a, "g" + a))
            .append(
                "if(weightg"
                    + a
                    + "!=0.0)value+=weightg"
                    + a
                    + "*v"
                    + (forward.inputs.length + a)
                    + "[j];}");
    } else if (kind == Kind.ONEHOT) {
      int classes = output.dim(output.rank() - 1);
      s.append(
          "value=v0[i/"
              + Math.max(1, classes)
              + "]==float(i%"
              + Math.max(1, classes)
              + ")?1.0:0.0;");
    } else if (kind == Kind.PROD) {
      int inner = inner(), width = inputs[0].dim(axis);
      s.append(
          "value=1.0;for(int k=0;k<"
              + width
              + ";k++)value*=v0[(i/"
              + Math.max(1, inner)
              + "*"
              + width
              + "+k)*"
              + inner
              + "+i%"
              + Math.max(1, inner)
              + "];");
    } else if (kind == Kind.SCATTER)
      s.append("for(int j=0;j<" + parameters.length + ";j++)if(ix[j]==i)value+=v0[j];");
    else if (kind == Kind.SEGMENT) {
      int inner = size(inputs[0]) / Math.max(1, inputs[0].dim(0));
      s.append(
          "for(int j=0;j<"
              + inputs[0].dim(0)
              + ";j++)if(v1[j]==float(i/"
              + Math.max(1, inner)
              + "))value+=v0[j*"
              + inner
              + "+i%"
              + Math.max(1, inner)
              + "];");
    } else {
      s.append(mapping("i", "f"));
      if (kind == Kind.TOPK && parameters[1] != 0)
        s.append("value=float(indexf%" + Math.max(1, inputs[0].dim(inputs[0].rank() - 1)) + ");");
      else {
        s.append(
            "if(indexf<0)value="
                + (kind == Kind.GATHER ? "0.0" : "uintBitsToFloat(0x7fc00000u)")
                + ";else {");
        for (int a = 0; a < inputs.length; a++)
          s.append("if(selectedf==" + a + ")value=v" + a + "[indexf];");
        s.append('}');
      }
    }
    return s.append("outValues[i]=value;").toString();
  }

  /** Scalar-array function used by Vulkan structured-region lowering. */
  public String asFunction(String name) {
    StringBuilder s =
        new StringBuilder(
            "void " + name + "(out float outValues[" + Math.max(1, size(output)) + "]");
    for (int a = 0; a < inputs.length; a++)
      s.append(",in float v" + a + "[" + Math.max(1, size(inputs[a])) + "]");
    s.append("){");
    ArrayKernel base = this;
    while (base.forward != null) base = base.forward;
    int[] map = base.parameters;
    Kind mapKind = base.kind;
    if (mapKind == Kind.GATHER || mapKind == Kind.SCATTER) {
      s.append("const int ix[" + Math.max(1, map.length) + "]=int[](");
      if (map.length == 0) s.append(-1);
      for (int j = 0; j < map.length; j++) {
        if (j > 0) s.append(',');
        s.append(map[j]);
      }
      s.append(");");
    }
    return s.append("for(int i=0;i<" + size(output) + ";i++){")
        .append(body())
        .append("}}\n")
        .toString();
  }

  public String source(boolean cuda) {
    StringBuilder s =
        new StringBuilder(
            cuda
                ? "__device__ float clamp(float x,float lo,float hi){return fminf(fmaxf(x,lo),hi);}"
                : "#version 450\nlayout(local_size_x=256) in;\n");
    ArrayKernel base = this;
    while (base.forward != null) base = base.forward;
    int[] map = base.parameters;
    Kind mapKind = base.kind;
    if (mapKind == Kind.GATHER || mapKind == Kind.SCATTER) {
      s.append(
          (cuda ? "__device__ int ix[" : "const int ix[")
              + Math.max(1, map.length)
              + "]="
              + (cuda ? "{" : "int[]("));
      if (map.length == 0) s.append(-1);
      for (int j = 0; j < map.length; j++) {
        if (j > 0) s.append(',');
        s.append(map[j]);
      }
      s.append(cuda ? "};" : " );");
    }
    if (cuda) {
      s.append("extern \"C\" __global__ void array_kernel(float* outValues");
      for (int a = 0; a < inputs.length; a++) s.append(",const float* v" + a);
      s.append("){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=" + size(output) + ")return;");
    } else {
      s.append("layout(std430,binding=0) writeonly buffer O{float outValues[];};");
      for (int a = 0; a < inputs.length; a++)
        s.append(
            "layout(std430,binding="
                + (a + 1)
                + ") readonly buffer I"
                + a
                + "{float v"
                + a
                + "[];};");
      s.append("void main(){int i=int(gl_GlobalInvocationID.x);if(i>=" + size(output) + ")return;");
    }
    s.append(body()).append('}');
    return cuda ? Elementwise.cuda(s.toString()) : s.toString();
  }
}
