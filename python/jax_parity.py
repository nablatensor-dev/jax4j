"""Regenerate deterministic F32 fixtures with JAX 0.11.2 on CPU only."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HIP_VISIBLE_DEVICES']=''
os.environ['ROCR_VISIBLE_DEVICES']=''
from pathlib import Path
import jax
import jax.numpy as jnp
if jax.__version__ != '0.11.2':
    raise RuntimeError(f'Expected JAX 0.11.2, found {jax.__version__}')
x=jnp.array([-.8,-.2,.3,.9],dtype=jnp.float32)
def loss(p):
    y=jnp.tanh(p*1.5)
    return jnp.sum(y*y)
value,grad=jax.jit(jax.value_and_grad(loss))(x)
rows=jnp.reshape(jnp.arange(1,7,dtype=jnp.float32),(2,3))
mapped=jax.jit(jax.vmap(jax.grad(lambda row:jnp.sum(row*row))))(rows)
lines=['jax.version='+jax.__version__,'platform=cpu']
for key,data in [('input',x),('value',jnp.atleast_1d(value)),('gradient',grad),('mappedGradient',mapped.reshape(-1))]:
    lines.append(key+'='+','.join(str(float(v)) for v in data.tolist()))
# Independent fixtures for the tensor transformations and device regions.
tangent=jnp.array([1,2,3,4],dtype=jnp.float32)
_,directional=jax.jvp(loss,(x,),(tangent,))
vector=lambda p:jnp.tanh(p*1.5)
scan_x=jnp.array([[1.],[2.],[3.]],dtype=jnp.float32)
scan_initial=jnp.array([2.],dtype=jnp.float32)
def scan(initial,values):
    def step(c,v):
        n=c*v
        return n,n
    return jax.lax.scan(step,initial,values)
scan_value,scan_outputs=scan(scan_initial,scan_x)
scan_grad_initial,scan_grad_x=jax.grad(lambda c,v:jnp.sum(scan(c,v)[0]),argnums=(0,1))(scan_initial,scan_x)
extra=[('tangent',tangent),('directional',jnp.atleast_1d(directional)),('hessian',jax.hessian(loss)(x)),('jacobian',jax.jacfwd(vector)(x)),('scanCarry',scan_value),('scanOutputs',scan_outputs),('scanGradInitial',scan_grad_initial),('scanGradInputs',scan_grad_x)]
for key,data in extra:
    lines.append(key+'='+','.join(str(float(v)) for v in data.reshape(-1).tolist()))
# Three complete functional Adam steps; no external optimizer package is needed.
@jax.jit
def train_step(p,m,v,step):
    g=jax.grad(lambda q:jnp.sum(q*q))(p)
    step=step+1
    m=.9*m+.1*g
    v=.999*v+.001*g*g
    p=p-.01*(m/(1-.9**step))/(jnp.sqrt(v/(1-.999**step))+1e-8)
    return p,m,v,step
p=jnp.array([1.,2.],dtype=jnp.float32)
m=jnp.zeros_like(p);v=jnp.zeros_like(p);step=jnp.array([0.],dtype=jnp.float32)
for _ in range(3):
    p,m,v,step=train_step(p,m,v,step)
lines.append('adamParameters='+','.join(str(float(q)) for q in p.tolist()))
text='\n'.join(lines)+'\n'
for destination in ['jax4j-core/src/test/resources/jax-parity.properties','jax4j-engine-vulkan/src/test/resources/jax-parity.properties','jax4j-engine-wgpu/src/test/resources/jax-parity.properties']:
    target=Path(destination)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(text)
