import torch

class Optimizable:
    '''
    This is the interface for anything that has parameters that need to be
    optimized, somewhat like torch.nn.Model but with the right plumbing for
    hyperoptimizability. (Specifically, torch.nn.Model uses the Parameter
    interface which does not give us enough control about the detachments.)
    Nominal operation of an Optimizable at the lowest level is as follows:
        o = MyOptimizable(...)
        o.initialize()
        loop {
            o.begin()
            o.zero_grad()
            loss = --compute loss function from parameters--
            loss.backward()
            o.step()
        }
    Optimizables recursively handle updates to their optimiz*ers*.
    '''
    def __init__(self, parameters, optimizer):
        self.parameters = parameters # a dict mapping names to tensors
        self.optimizer = optimizer   # which must itself be Optimizable!
        self.all_params_with_gradients = []

    def initialize(self):
        ''' Initialize parameters, e.g. with a Kaiming initializer. '''
        pass
    
    def begin(self):
        ''' Enable gradient tracking on current parameters. '''
        for param in self.all_params_with_gradients:
             param.grad = None
        self.all_params_with_gradients.clear()
        for name, param in self.parameters.items():
            param.requires_grad_() # keep gradient information...
            param.retain_grad()    # even if not a leaf...
            self.all_params_with_gradients.append(param)
        self.optimizer.begin()

    def zero_grad(self):
        ''' Set all gradients to zero. '''
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)
        self.optimizer.zero_grad()

    ''' Note: at this point you would probably call .backwards() on the loss
    function. '''

    def step(self):
        ''' Update parameters '''
        pass

class NoOpOptimizer(Optimizable):
    '''
    NoOpOptimizer sits on top of a stack, and does not affect what lies below.
    '''
    def __init__(self):
        pass

    def initialize(self):
        pass

    def begin(self):
        pass

    def zero_grad(self):
        pass

    def step(self, params):
        pass

    def __str__(self):
        return ''

class SGD(Optimizable):
    '''
    A hyperoptimizable SGD.
    '''
    def __init__(self, alpha=0.01, mu=0.0, optimizer=NoOpOptimizer()):
        self.mu = mu
        self.state = {}
        parameters = {
            'alpha': torch.tensor(alpha),
            'mu': torch.tensor(mu)
        }
        super().__init__(parameters, optimizer)

    def step(self, params):
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            g = param.grad.detach()
            p = param.detach()
            if self.mu != 0.0:
                if name not in self.state:
                    buf = self.state[name] = g
                else:
                    buf = self.state[name].detach()
                    buf = buf * self.parameters['mu'] + g
                g = self.state[name] = buf
            params[name] = p - g * self.parameters['alpha']
        
    def __str__(self):
        return 'sgd / '+ str(self.optimizer)

class SGDPerParam(Optimizable):
    '''
    Optimizes parameters individually with SGD.
    '''
    def __init__(self, params, optimizer=NoOpOptimizer()):
        parameters = {k + '_alpha' : torch.tensor(v) for k, v in params}
        super().__init__(parameters, optimizer)

    def step(self, params):
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            g = param.grad.detach()
            p = param.detach()
            if name + '_alpha' not in self.parameters: params[name] = p
            else: params[name] = p - g * self.parameters[name + '_alpha']

    def __str__(self):
        return 'sgdPerParam / ' + str(self.optimizer)

class AdaGrad(Optimizable):
    '''
    A hyperoptimizable AdaGrad.
    '''
    def __init__(self, alpha=0.01, optimizer=NoOpOptimizer()):
        self.eps = 1e-10
        self.cache = {}
        parameters = {
            'alpha': torch.tensor(alpha)
        }
        super().__init__(parameters, optimizer)
    
    def step(self, params):
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    'G': torch.zeros_like(param) + 1e-1
                }
            g = param.grad.detach()
            self.cache[name]['G'] = G = self.cache[name]['G'].detach() + torch.square(g)
            params[name] = param.detach() - self.parameters['alpha'] * g / torch.sqrt(G + self.eps).detach()
    
    def __str__(self):
        return 'adagrad / ' + str(self.optimizer)

class RMSProp(Optimizable):
    '''
    A hyperoptimizable RMSProp.
    '''
    def clamp(x):
        return (x.tanh() + 1.) / 2.

    def unclamp(y):
        z = y * 2. - 1.
        return ((1. + z) / (1. - z)).log() / 2.

    def __init__(self, alpha=0.01, gamma=0.99, optimizer=NoOpOptimizer()):
        self.eps = 1e-8
        parameters = {
            'alpha': torch.sqrt(torch.tensor(alpha)),
            'gamma': RMSProp.unclamp(torch.tensor(gamma))
        }
        super().__init__(parameters, optimizer)
        self.cache = {}

    def step(self, params):
        self.optimizer.step(self.parameters)
        gamma = RMSProp.clamp(self.parameters['gamma'])
        alpha = torch.square(self.parameters['alpha'])
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    's': torch.zeros_like(param)
                }
            g = param.grad.detach()
            self.cache[name]['s'] = s = gamma * self.cache[name]['s'].detach() + (1. - gamma) * torch.square(g)
            self.all_params_with_gradients.append(s)
            params[name] = param.detach() - alpha * g / torch.sqrt(s + self.eps)
    
    def __str__(self):
        return 'rmsprop / ' + str(self.optimizer)

class RMSPropAlpha(Optimizable):
    '''
    A hyperoptimizable RMSProp for only alpha.
    '''
    def __init__(self, alpha=0.01, gamma=0.99, optimizer=NoOpOptimizer()):
        self.eps = 1e-8
        self.gamma = gamma
        parameters = {
            'alpha': torch.sqrt(torch.tensor(alpha)),
        }
        super().__init__(parameters, optimizer)
        self.cache = {}

    def step(self, params):
        self.optimizer.step(self.parameters)
        alpha = torch.square(self.parameters['alpha'])
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    's': torch.zeros_like(param)
                }
            g = param.grad.detach()
            self.cache[name]['s'] = s = self.gamma * self.cache[name]['s'].detach() + (1. - self.gamma) * torch.square(g)
            self.all_params_with_gradients.append(s)
            params[name] = param.detach() - alpha * g / torch.sqrt(s + self.eps)
    
    def __str__(self):
        return 'rmspropAlpha / ' + str(self.optimizer)

class Adam(Optimizable):
    '''
    A hyperoptimizable Adam optimizer.
    '''
    def clamp(x):
        return (x.tanh() + 1.) / 2.

    def unclamp(y):
        z = y * 2. - 1.
        return ((1. + z) / (1. - z)).log() / 2.

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, log_eps=-8., optimizer=NoOpOptimizer()):
        self.eps = 10. ** log_eps
        parameters = {
            'alpha': torch.tensor(alpha),
            'beta1': Adam.unclamp(torch.tensor(beta1)),
            'beta2': Adam.unclamp(torch.tensor(beta2)),
        }
        super().__init__(parameters, optimizer)
        self.num_stepments = 0
        self.cache = {}

    def step(self, params):
        self.num_stepments += 1
        self.optimizer.step(self.parameters)
        t = self.num_stepments
        beta1 = Adam.clamp(self.parameters['beta1'])
        beta2 = Adam.clamp(self.parameters['beta2'])
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    'm': torch.zeros_like(param),
                    'v': torch.zeros_like(param) +\
                            self.eps
# NOTE that we add a little `fudge factor' here because sqrt is not
# differentiable at exactly zero
                }
            g = param.grad.detach()
            self.cache[name]['m'] = m =\
                beta1 * self.cache[name]['m'].detach() + (1. - beta1) * g
            self.cache[name]['v'] = v =\
                beta2 * self.cache[name]['v'].detach() + (1. - beta2) * g * g
            self.all_params_with_gradients.append(m)
            self.all_params_with_gradients.append(v)

            m_hat = m / (1. - beta1 ** float(t))
            v_hat = v / (1. - beta2 ** float(t))

            dparam = m_hat / (v_hat ** 0.5 + self.eps)
            params[name] = param.detach() - self.parameters['alpha'] * dparam

    def __str__(self):
        return 'adam / ' + str(self.optimizer)

class AdamBaydin(Optimizable):
    ''' Same as above, but only optimizes the learning rate, treating the
    remaining hyperparameters as constants. '''

    def __init__(
        self,
        alpha=0.001, beta1=0.9, beta2=0.999, log_eps=-8.,
        optimizer=NoOpOptimizer()
    ):
        parameters = {
            'alpha': torch.tensor(alpha),
        }
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.log_eps = log_eps
        super().__init__(parameters, optimizer)
        self.num_stepments = 0
        self.cache = {}

    def step(self, params):
        self.num_stepments += 1
        self.optimizer.step(self.parameters)
        t = self.num_stepments
        beta1 = self.beta1
        beta2 = self.beta2
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    'm': torch.zeros_like(param),
                    'v': torch.zeros_like(param) +\
                            10.**self.log_eps
# NOTE that we add a little `fudge factor' here because sqrt is not
# differentiable at exactly zero
                }

            g = param.grad.detach()
            self.cache[name]['m'] = m =\
                beta1 * self.cache[name]['m'].detach() + (1. - beta1) * g
            self.cache[name]['v'] = v =\
                beta2 * self.cache[name]['v'].detach() + (1. - beta2) * g * g

            self.all_params_with_gradients.append(m)
            self.all_params_with_gradients.append(v)

            m_hat = m / (1. - beta1 ** float(t))
            v_hat = v / (1. - beta2 ** float(t))

            dparam = m_hat / (v_hat ** 0.5 + 10. ** self.log_eps)
            params[name] = param.detach() - self.parameters['alpha'] * dparam

    def __str__(self):
        return 'adamBaydin / ' + str(self.optimizer)


class ModuleWrapper(Optimizable):
    '''
    This class tries to convert a torch.nn.Module to an Optimizable, handling
    the internal plumbing needed to update parameters correctly.
    '''
    def __init__(self, module, optimizer=NoOpOptimizer()):
        self.module = module
        parameters = {k:v for k, v in module.named_parameters(recurse=True)}
        super().__init__(parameters, optimizer)
    
    def initialize(self):
        self.optimizer.initialize()
    
    def zero_grad(self):
        """ Set all gradients to zero. """
        self.module.zero_grad()
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)
        self.optimizer.zero_grad()
    
    def forward(self, *xyz):
        return self.module(*xyz)
    
    def train(self):
        self.module.train()
    
    def eval(self):
        self.module.eval()
    
    def step(self):
        self.optimizer.step(self.parameters)
        def set_param(m, k, v):
            kk = k
            while '.' in k:
                sm = k[:k.index('.')]
                k = k[k.index('.') + 1:]
                m = m._modules[sm]

            m._parameters[k] = None
            m._parameters[k] = self.parameters[kk]

        for k, v in self.module.named_parameters(recurse=True):
            set_param(self.module, k, v)