# coding=utf-8
from __future__ import absolute_import, division, print_function
def outer_factory():
    self = None
    step_function = None

    def inner_factory(ag__):

        def tf__predict_function(iterator):
            'Runs an evaluation execution with one step.'
            with ag__.FunctionScope('predict_function', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                try:
                    do_return = True
                    retval_ = ag__.converted_call(ag__.ld(step_function), (ag__.ld(self), ag__.ld(iterator)), None, fscope)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__predict_function
    return inner_factory