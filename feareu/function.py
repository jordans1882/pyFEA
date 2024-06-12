import numpy as np

# from FEA.optimizationproblems.benchmarks import rastrigin__


class Function:
    def __init__(self, context, function, factor):
        self.function = function
        self.context = np.copy(context)
        self.factor = factor

    def __call__(self, arg):
        """print("context shape: ", self.context.shape)
        print("context: ", self.context)
        print("factor len: ", len(self.factor))
        print("factor: ", self.factor)
        print("args shape: ", len(arg))
        print("args: ", arg)"""
        self.context[self.factor] = arg
        return self.function(self.context)


#     def _construct_wrapper_fun(partial_ctx, remaining_context):
#         self.function(concat(partial_ctx, remaining_ctx))
#         self.function(self.context)
"""def main():
    fun = Function([1, 2, 0, 0, 0], rastrigin__, [0, 1])
    print(fun([0, 0]))
if __name__ == "__main__":
    main()"""
