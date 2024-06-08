import torch.optim

from typing import List, Optional, Union, Tuple,Dict
import logging
import torch
from torch import Tensor
from torch.optim.optimizer import (ParamsT, _use_grad_for_differentiable, _get_value,
                                   _dispatch_sqrt, _default_to_fused_or_foreach)

from pytorch_ps.ps.client import PSClient

adam_logger = logging.getLogger("hash_sparse_adam")
class HashSparseAdam(torch.optim.Adam):

    def __init__(self,
                 ps_client:PSClient,
                 param_names:List[str],
                 params: ParamsT,
                 lr: Union[float, Tensor] = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 amsgrad: bool = False,
                 *,
                 foreach: Optional[bool] = None,
                 maximize: bool = False,
                 capturable: bool = False,
                 differentiable: bool = False,
                 fused: Optional[bool] = None):
        """ embedding variable names"""
        self.param_names = param_names
        """cache embedding indices in context"""
        self.select_index:Dict[str,List] = {}
        self.ps_client = ps_client
        super(HashSparseAdam, self).__init__(params,
                                         lr,
                                         betas,
                                         eps,
                                         weight_decay,
                                         amsgrad,
                                         foreach=foreach,
                                         maximize=maximize,
                                         capturable=capturable,
                                         differentiable=differentiable,
                                         fused=fused)

    def _init_group(
            self,
            group,
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps
    ):
        has_complex = False
        """clear context"""
        self.select_index.clear()

        for p,p_name in zip(group['params'],self.param_names):
            adam_logger.info(f"`init_group` parameter:{p_name } ,size: {p.size()}")
            if not p.grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

            if p.grad is not None:
                has_complex |= torch.is_complex(p)

                dim = p.grad.size(dim=1)
                grad_indices = p.grad._indices().tolist()
                if adam_logger.isEnabledFor(logging.DEBUG):
                    adam_logger.debug(f"{p_name} grad_indices:{grad_indices}")
                dup_set = set()
                tmp_select_ind = []
                for v in grad_indices[0]:
                    if v in dup_set:
                        continue
                    dup_set.add(v)
                    tmp_select_ind.append(v)
                self.select_index[p_name] = tmp_select_ind
                grad = torch.index_select(input=p.grad, dim=0,
                                          index=torch.tensor(tmp_select_ind, dtype=torch.long)).to_dense()
                output = self.ps_client._get_emb_context(p_name)
                if adam_logger.isEnabledFor(logging.DEBUG):
                    adam_logger.debug(f"context: {p_name} select_index:{tmp_select_ind}")
                    adam_logger.debug(f"sparse adam grad:{grad}")
                    adam_logger.debug(f"before emb output:{output}")

                params_with_grad.append(output)
                grads.append(grad)



                state = self.ps_client.get_training_state(p_name,tmp_select_ind,dim)

                exp_avg = state["exp_avg"]
                exp_sp_avg = state["exp_sp_avg"]

                exp_avgs.append(exp_avg)
                exp_avg_sqs.append(exp_sp_avg)
                state_steps.append(state['step'])
                if adam_logger.isEnabledFor(logging.DEBUG):
                    adam_logger.debug(f"{p_name} before step :{state['step']}")
                    adam_logger.debug(f"{p_name} before step exp_avg:{exp_avg}")
                    adam_logger.debug(f"{p_name} before step exp_sp_avg{exp_sp_avg}")

                if group['amsgrad']:
                    raise RuntimeError('`amsgrad` is not supported for `step`')
                if group['differentiable'] and state['step'].requires_grad:
                    raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')

                # Foreach without capturable does not support a tensor lr
                if group['foreach'] and torch.is_tensor(group['lr']) and not group['capturable']:
                    raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True')


        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

                Args:
                    closure (Callable, optional): A closure that reevaluates the model
                        and returns the loss.
                """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if len(self.param_groups) > 1:
            raise RuntimeError("`param_groups` more than one"+len(self.param_groups),self.param_groups.keys())

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                foreach=group['foreach'],
                capturable=group['capturable'],
                differentiable=group['differentiable'],
                fused=group['fused'],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

            adam_logger.debug("********* updated grad ********** ")
            for p_name,grad,exp_avg,exp_avg_sq,state_step in zip(self.param_names,grads,exp_avgs,exp_avg_sqs,state_steps):
                if adam_logger.isEnabledFor(logging.DEBUG):
                    adam_logger.debug(f"{p_name},grad:{grad}")
                self.ps_client.set_training_state(p_name,self.select_index[p_name],exp_avg,exp_avg_sq,state_step)

                param_state = self.ps_client.get_training_state(p_name, self.select_index[p_name],dim=grad.size(-1))


                if adam_logger.isEnabledFor(logging.DEBUG):
                    exp_equal_value = torch.sum((torch.sub(exp_avg, param_state["exp_avg"]) < 0.00001).to(dtype=torch.int)).item()
                    adam_logger.debug(f"exp_avg equals:{exp_equal_value}=={exp_avg.numel()}")
                    exp_sq_equal_value = torch.sum((torch.sub(exp_avg_sq, param_state["exp_sp_avg"])<0.00001).to(dtype=torch.int))
                    adam_logger.debug(f"exp_avg equals:{exp_sq_equal_value}=={exp_avg_sq.numel()}")

                """ set embedding layer out where  embedding forward """
                output = self.ps_client._get_emb_context(p_name)

                if adam_logger.isEnabledFor(logging.DEBUG):
                    adam_logger.debug(f"{p_name}  output:{output}")

                self.ps_client.update_tensor_to_dic(p_name,self.select_index[p_name],output)

        """clear context"""
        self.ps_client._clear_emb_context()
        self.select_index.clear()

        return loss



"""
The following content is just copied from
torch.optim.adam.py 

"""

def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[Tensor],
         # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
         # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
         foreach: Optional[bool] = None,
         capturable: bool = False,
         differentiable: bool = False,
         fused: Optional[bool] = None,
         grad_scale: Optional[Tensor] = None,
         found_inf: Optional[Tensor] = None,
         has_complex: bool = False,
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: Union[float, Tensor],
         weight_decay: float,
         eps: float,
         maximize: bool):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """
    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch.jit.is_scripting():
        raise RuntimeError("not supported with fused optimizers")
        #func = _fused_adam
    elif foreach and not torch.jit.is_scripting():
        raise RuntimeError("not supported with foreach optimizers")
        #func = _multi_tensor_adam
    else:
        func = _single_tensor_adam

    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad=amsgrad,
         has_complex=has_complex,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         capturable=capturable,
         differentiable=differentiable,
         grad_scale=grad_scale,
         found_inf=found_inf)


def _single_tensor_adam(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        state_steps: List[Tensor],
                        grad_scale: Optional[Tensor],
                        found_inf: Optional[Tensor],
                        *,
                        amsgrad: bool,
                        has_complex: bool,
                        beta1: float,
                        beta2: float,
                        lr: Union[float, Tensor],
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        capturable: bool,
                        differentiable: bool):

    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch._utils.is_compiling() and capturable:
            assert (
                (param.is_cuda and step_t.is_cuda) or (param.is_xla and step_t.is_xla)
            ), "If capturable=True, params and state_steps must be CUDA or XLA tensors."

        # update step
        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        if capturable or differentiable:
            step = step_t

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size)

        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])


