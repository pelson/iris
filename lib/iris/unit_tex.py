# -*- coding: utf-8 -*-
from collections import namedtuple
import pyparsing
import itertools


class Operand(object):
    def __init__(self, obj):
        self.op_value = obj

    def __str__(self):
        return '{}'.format(self.op_value)

    def __repr__(self):
        return '{!r}'.format(self.op_value)

    def UDUNIT2_form(self):
        return str(self)

    def latex(self, parent=None):
        if isinstance(self.op_value, basestring):
            operand = r'\mathrm{{{}}}'.format(self.op_value)
        elif hasattr(self.op_value, 'latex'):
            operand = self.op_value.latex()
        else:
            operand = '{{{}}}'.format(self.op_value)
        return operand

    def walk_lhs(self):
        yield self

    def walk_rhs(self):
        yield self


class BinaryOp(namedtuple('BinaryOp', ['lhs', 'rhs'])):
    def _binary_op_tex(self):
        lhs = self.lhs.latex(parent=self)
        rhs = self.rhs.latex(parent=self)
        return lhs, rhs

    def walk_lhs(self):
        return itertools.chain([self], self.lhs.walk_lhs())

    def walk_rhs(self):
        return itertools.chain([self], self.rhs.walk_rhs())

    @property
    def op_value(self):
        """
        Implement the interface provided by :class:`Operand`
        by returning self.

        """
        return self


class Division(BinaryOp):
    def __str__(self):
        return '({s.lhs} / {s.rhs})'.format(s=self)

    def __repr__(self):
        return '([{s.lhs!r}, "/", {s.rhs!r}])'.format(s=self)

    def UDUNIT2_form(self):
        return '({}/{})'.format(self.lhs.UDUNIT2_form(),
                                self.rhs.UDUNIT2_form())

    def latex(self, parent=None):
        lhs, rhs = self._binary_op_tex()
        # Note: PEP 3101 allows you to escape "{" by doubling it up.
        return r'\frac{{{lhs}}}{{{rhs}}}'.format(lhs=lhs, rhs=rhs)


class Power(BinaryOp):
    def __str__(self):
        return '({s.lhs} ^ {s.rhs})'.format(s=self)

    def __repr__(self):
        return '([{s.lhs!r}, "^", {s.rhs!r}])'.format(s=self)

    def UDUNIT2_form(self):
        return '({}**{})'.format(self.lhs.UDUNIT2_form(),
                                 self.rhs.UDUNIT2_form())

    def latex(self, parent=None):
        lhs, rhs = self._binary_op_tex()

        lhs_op = self.lhs
        if isinstance(lhs_op, Operand):
            lhs_op = lhs_op.op_value

        # Raising a negative number requires that the number has brackets.
        if (isinstance(parent, BinaryOp) and self is parent.rhs and
                isinstance(lhs_op, (int, float)) and lhs_op < 0):
            lhs = r'({lhs})'.format(lhs=lhs)

        return r'{lhs}^{rhs}'.format(lhs=lhs, rhs=rhs)


class Multiplication(BinaryOp):
    def __str__(self):
        return '({s.lhs} * {s.rhs})'.format(s=self)

    def __repr__(self):
        return '([{s.lhs!r}, "*", {s.rhs!r}])'.format(s=self)

    def UDUNIT2_form(self):
        return '({}*{})'.format(self.lhs.UDUNIT2_form(),
                                self.rhs.UDUNIT2_form())

    def _binary_op_tex(self):
        lhs, rhs = BinaryOp._binary_op_tex(self)

        lhs_op, rhs_op = self.lhs, self.rhs
        if isinstance(lhs_op, Operand):
            lhs_op = lhs_op.op_value

        if isinstance(rhs_op, Operand):
            rhs_op = rhs_op.op_value

        # Multiplication with a negative number
        # requires that the number has brackets.
        if isinstance(lhs_op, (int, float)) and lhs_op < 0:
            lhs = r'({lhs})'.format(lhs=lhs)

        if isinstance(rhs_op, (int, float)) and rhs_op < 0:
            rhs = r'({rhs})'.format(rhs=rhs)

        return lhs, rhs

    def latex(self, parent=None):
        mult_template = r'{lhs}{rhs}'

        # Figure out some context about this Multiplication:
        #  * Are all the right hand side operands after self.lhs Multiplication
        #    instances?
        #  * Are all the right hand side operands after
        #    self.rhs (Multiplication or Power) instances?
        #  * Get hold of the Operands at the extreme left of self.rhs and
        #    the extreme right of self.lhs
        rhs_of_self_lhs = list(self.lhs.walk_rhs())
        all_lhs_mult = all([isinstance(v, Multiplication)
                            for v in rhs_of_self_lhs[:-1]])
        extreme_rhs_of_self_lhs = rhs_of_self_lhs[-1].op_value

        lhs_of_self_rhs = list(self.rhs.walk_lhs())
        all_lhs_mult_or_pow = all([isinstance(v, (Multiplication, Power))
                                   for v in lhs_of_self_rhs[:-1]])
        extreme_lhs_of_self_rhs = lhs_of_self_rhs[-1].op_value

        # Figure out if the the extreme lhs of self.rhs is a positive number
        lhs_of_self_rhs_positive = (isinstance(extreme_lhs_of_self_rhs,
                                               (int, float))
                                    and extreme_lhs_of_self_rhs >= 0)

        # Figure out if the the extreme rhs of self.lhs is a positive number
        rhs_of_self_lhs_positive = (isinstance(extreme_rhs_of_self_lhs,
                                               (int, float))
                                    and extreme_rhs_of_self_lhs >= 0)

        # We may need to separate the operands if all the RHSides of self.lhs
        # are Multiplications and all the LHSides of self.rhs are either
        # Power or Multiplications
        if all_lhs_mult and all_lhs_mult_or_pow:

            # If we have positive numbers (i.e. there are no brackets around
            # them), we need to put in a times sign.
            if lhs_of_self_rhs_positive and rhs_of_self_lhs_positive:
                mult_template = r'{lhs}\times{rhs}'

            # Alternatively, if the RHS is a variable, put in a space
            elif isinstance(extreme_rhs_of_self_lhs, basestring):
                mult_template = r'{lhs}\ {rhs}'

        lhs, rhs = self._binary_op_tex()
        return mult_template.format(lhs=lhs, rhs=rhs)


class UnitParser(object):
    """
    The class which parses UDUNITS input strings, and converts into
    a nested representation of BinaryOp and Operand instances.

    The primary interface is the :meth:`.parse` method.

    """
    DEBUG = False

    def _debug_msg(self, msg):
        if self.DEBUG:
            print msg

    #: Tuple of the the valid (caseless) exponential operators.
    EXP_OPERATORS = ('^', '**', '+', '-')

    #: Tuple of the valid (caseless) multiplication operators.
    MULT_OPERATORS = ('*', '/', '.', '+', ' +', '·')

    #: Tuple of the valid (caseless) division operators.
    DIV_OPERATORS = ('/', ' per ')

    #: Tuple of all valid (caseless) operators.
    ALL_OPERATORS = EXP_OPERATORS + MULT_OPERATORS + DIV_OPERATORS

    pp_float = pyparsing.Combine(
        pyparsing.ZeroOrMore('-') +
        pyparsing.Word(pyparsing.nums) +
        pyparsing.Optional('.' +
                           pyparsing.Optional(
                           pyparsing.Word(pyparsing.nums))) +
        pyparsing.Optional(
            pyparsing.oneOf("e E") +
            pyparsing.Optional(pyparsing.oneOf("+ -")) +
            pyparsing.Optional(pyparsing.Word(pyparsing.nums))
        )
    ).setName('potential float')

    _WORD = pyparsing.Word(pyparsing.alphas)

    def parse(self, unit_string):
        """
        Take the given ``unit_string`` and parse it into nested
        :class:`BinaryOp` instances.

        """
        # Force the unit string to have at least one set of brackets.
        unit_string = '({})'.format(unit_string)

        # Since there is no subtraction concept in udunits, capture numbers
        # with their leading "-" if they exist.
        pp_int = pyparsing.Combine(pyparsing.ZeroOrMore(
            pyparsing.oneOf(['-'])) +
            pyparsing.Word(pyparsing.nums) +
            pyparsing.NotAny(
                pyparsing.oneOf(['.', 'e', 'E']))
        )
        pp_int.setParseAction(lambda t: self._to_int(t[0]))

        pp_float = pyparsing.Combine(
            pyparsing.ZeroOrMore(pyparsing.oneOf(['-'])) +
            pyparsing.Word(pyparsing.nums) +
            pyparsing.Optional('.' +
                               pyparsing.Optional(
                               pyparsing.Word(pyparsing.nums))) +
            pyparsing.Optional(
                pyparsing.oneOf("e E") +
                pyparsing.Optional(pyparsing.oneOf("+ -")) +
                pyparsing.Optional(
                    pyparsing.Word(pyparsing.nums))
            )
        )

        # Define the contents inside the brackets. If there exists contents
        # which are not defined here, an exception will ensue.
        bracket_content = (self._WORD | pp_int | pp_float |
                           pyparsing.oneOf(self.ALL_OPERATORS,
                                           caseless=True) |
                           pyparsing.White()).leaveWhitespace()

        group_by_parens = pyparsing.nestedExpr(opener='(', closer=')',
                                               content=bracket_content)
        # define the function which gets called when a group is found.
        group_by_parens.parseAction = [self._pyparsing_action]

        # Parse the unit_string, returning the processed group into a
        # pyparsing.ParseResults instance.
        parsed_groups = group_by_parens.parseString(unit_string)

        # Something went really wrong (I don't know how this can happen).
        assert len(parsed_groups) == 1
        # Return the top level BinaryOp.
        return parsed_groups[0]

    def iter_tripple(self, iterable):
        "s -> (s0,s1,s2), (s1,s2,s3), (s2,s3,s4), ..."
        a, b, c = itertools.tee(iterable, 3)
        next(b, None)
        next(c, None)
        next(c, None)
        return itertools.izip(a, b, c)

    def iter_pair(self, iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return itertools.izip(a, b)

    def _pyparsing_action(self, s, locs, toks):
        """The action which runs each time a parse group is found."""
        r = self.lexed_group_to_op_tree(toks[0])
        toks[0] = r
        return toks

    @staticmethod
    def _to_int(val):
        """
        Turns a string such as '1' into the int 1.
        Also handles the case of '--1' to 1.

        """
        val = str(val).replace(' ', '')
        while val.startswith('--'):
            val = val[2:]
        return int(val)

    @staticmethod
    def _to_float(val):
        """
        Turns a string such as '1e2' into the float 100.
        Also handles the case of '--1e2' to 100.

        """
        val = str(val).replace(' ', '')
        while val.startswith('--'):
            val = val[2:]
        return float(val)

    def is_float_like(self, operand):
        return (isinstance(operand.op_value, basestring)
                and self.pp_float.searchString(operand.op_value))

    def is_variable_like(self, operand):
        return (isinstance(operand.op_value, basestring)
                and not self.pp_float.searchString(operand.op_value))

    def valid_operand(self, value):
        """
        Return whether the given value is a valid operand.
        Compliment of :meth:`.valid_operator`.

        """
        return not self.valid_operator(value)

    def valid_operator(self, value):
        """
        Return whether the given value is a valid operator by checking whether
        value is in :data:`.ALL_OPERATORS`.

        """
        return (isinstance(value, basestring) and value.strip() and
                value.lower() in (self.ALL_OPERATORS))

    def _normalized_whitespace(self, group):
        """
        Removes whitespace from a lexed group.

        >>> group = [1, '*', '-2.', '*', '+', '3.']
        >>> UnitParser()._normalized_whitespace(group)
        [1, '*', '-2.', '*', '3.']

        """
        if len(group) > 2:
            # We always add the first and last elements of group to the
            # normalized form
            new_group = [group[0]]
            for lh, middle, rh in self.iter_tripple(group):
                if (isinstance(middle, basestring) and not middle.strip()):
                    # If the middle value is an empty string
                    # and neither the left or right hand is a valid operator,
                    # then we must insert a multiplication.
                    if not (self.valid_operator(rh) or
                            self.valid_operator(lh)):
                        new_group.append('*')
                else:
                    # If the middle is a non-empty string and the left hand
                    # is a valid operator, then we should remember to add the
                    # middle.
                    if not (middle == '+' and self.valid_operator(lh)):
                        new_group.append(middle)
            else:
                new_group.append(rh)
            group = new_group
            self._debug_msg(
                'Normalised whitespace (phase 1): {}'.format(group))
        return group

    def _normalize_operands(self, group):
        """
        Removes operands next to other operands.

        """
        # Get rid of operands directly next to other operators
        # (such as s2 and 2s which translate to s^2 and 2*s respectively).
        if len(group) >= 2:
            new_r = []
            for lh, rh in self.iter_pair(group):
                # Normalize to single spaces
                if isinstance(lh, basestring) and not lh.strip():
                    lh = ' '

                new_r.append(lh)

                if self.valid_operand(lh) and self.valid_operand(rh):
                    lh_numeric = (isinstance(lh, int)
                                  or (isinstance(lh, basestring)
                                      and self.pp_float.searchString(lh)))
                    rh_numeric = (isinstance(rh, int)
                                  or (isinstance(rh, basestring)
                                      and self.pp_float.searchString(rh)))

                    if (self._WORD.searchString(lh)
                            and not self._WORD.searchString(rh)):
                        new_r.append('^')
                    elif lh_numeric and rh_numeric:
                        new_r.append('^')
                    else:
                        new_r.append('*')
            else:
                new_r.append(rh)
            group = new_r
            self._debug_msg('Normalised operands (phase 2): {}'.format(group))
        return group

    def _normalize_signs(self, group):
        """
        Removes leading sign operators.

        >>> group = ['-', 's', '^', 2]
        >>> UnitParser()._normalize_signs(group)
        ['-s', '^', 2]

        """
        sign = 1
        while group[0] in ['-', '+']:
            sign *= int(group.pop(0) + '1')
        if sign == -1:
            g0 = group.pop(0)
            if isinstance(g0, Operand):
                g0 = g0.op_value

            if isinstance(g0, basestring):
                g0 = '-' + g0
            else:
                g0 = g0 * sign
            group.insert(0, g0)

        if group[-1] in ['-', '+']:
            raise ValueError('Group had a trailing +/-. {}'.format(group))

        self._debug_msg('Normalised signs (phase 3): {}'.format(group))

        return group

    def lexed_group_to_op_tree(self, group):
        """
        Turn the lexed groups (the parts inside the brackets) into nested
        BinaryOp and Operand instances.

        .. important::

            Because of the syntax that UDUNITS supports, it is not possible to
            reliably identify floats or even operators before the ast step.
            This means that strings which look like numbers will be turned
            floats by this method.


        Example:

        >>> group = ['b', '  ', 2, '+', '2', '**', -1]
        >>> UnitParser().lexed_group_to_op_tree(group)
        ([(['b', "*", 2]), "*", ([2.0, "^", -1])])

        .. note::
            Despite the repr, the result really is a BinaryOp instance,
            with each of the leaves being Operand instances.

        """
        self._debug_msg('Input group: {}'.format(group))

        # Strip leading and trailing empty strings in the lexed group.
        while isinstance(group[0], basestring) and not group[0].strip():
            group = group[1:]
        while isinstance(group[-1], basestring) and not group[-1].strip():
            group = group[:-1]

        if not group:
            raise ValueError('The input group must not be of length 0.')

        group = self._normalized_whitespace(group)
        group = self._normalize_operands(group)
        group = self._normalize_signs(group)

        # If there was only one element inside the group, just return it
        # as an Operand - There are no BinaryOps to be made.
        if len(group) == 1:
            operand = group[0]

            if not isinstance(operand, Operand):
                operand = Operand(operand)

            # If the given group item is a string which looks
            # like a float, turn it into a float.
            if self.is_float_like(operand):
                operand = Operand(self._to_float(operand))

            group = [operand]

        ######################################################################
        # The meat of this method: Uses the mutability of "group" to iterate #
        # over the lexed items and replaces them with appropriate BinaryOp   #
        # or Operator instances. This is a multi-pass operation, with all of #
        # the Power operations needing to occur before the Multiplication    #
        # ones.                                                              #
        ######################################################################
        # Create a list of the operator types to handle. This list's
        # mutability is utilised to trigger another pass of the EXP
        # operators in the following while loop.
        remaining_op_groups = [self.EXP_OPERATORS,
                               (self.MULT_OPERATORS + self.DIV_OPERATORS)]

        while remaining_op_groups:
            op_set = remaining_op_groups.pop(0)
            is_exp = op_set is self.EXP_OPERATORS

            # Parts of the following for loop may be jumped / fast-forwarded
            # by sections inside the loop.
            jump = 0

            # We're making use of mutability and index-ability of "group", so
            # keep a track of difference in the number of items +/- original
            # length that have been added/removed from "group".
            offset = 0

            # Loop over the whole length of group such that we can get
            # (lhs, middle, rhs) triples via indexing.
            for i in xrange(len(group) - 2):

                # Fast forward the loop until jump == 0.
                if jump > 0:
                    jump -= 1
                    continue

                c = group[i + offset:i + offset + 3]
                lhs, middle, rhs = group[i + offset:i + offset + 3]

                # The situation should be that lhs and rhs are valid operands
                # and middle is a valid operator. In which case, the next loop
                # should be skipped (as then the lhs will be an operator)
                assert (self.valid_operand(lhs) and self.valid_operand(rhs))
                assert self.valid_operator(middle)
                jump = 1

                # If this operation is not for this pass, skip it
                if middle not in op_set:
                    continue

                # Remove lhs, middle and rhs from the list. Some operations
                # inside the loop may add them back again before continuing
                # the loop.
                group.pop(i + 2 + offset)
                group.pop(i + 1 + offset)
                group.pop(i + offset)

                lh_op = lhs
                if not isinstance(lhs, (BinaryOp, Operand)):
                    lh_op = Operand(lhs)

                rh_op = rhs
                if not isinstance(rhs, (BinaryOp, Operand)):
                    rh_op = Operand(rhs)

                if self.is_float_like(lh_op):
                    lhs = self._to_float(lhs)
                    lh_op = Operand(lhs)

                # There are several special cases about Exponentiation with
                # UDUNITS. They all involve floats on the rhs. If we meet any
                # of these cases, we need to fix the group up, and re-run the
                # loop with the exponential operator list:
                #    2.5^3.6 == 2.5**3.6 == (2.5 ^ 3) * 0.6
                #    2.5+3.6 == (2.5 * 3.6)
                #    s2.3 == (s ^ 2) * 3
                if is_exp and self.is_float_like(rh_op):
                    if self.is_variable_like(lh_op):
                        # Split the float into two operations (1.2 => 1 x 2)
                        rhs, new_part = rh_op.op_value.split('.')
                        rh_op = Operand(self._to_int(rhs))

                        # For genuine floats, we need to restart the for-loop
                        if new_part:
                            new_part = self._to_int(new_part)
                            group.insert(i + offset, new_part)
                            group.insert(i + offset, '*')
                            group.insert(i + offset, rh_op)
                            group.insert(i + offset, middle)
                            group.insert(i + offset, lh_op)

                            remaining_op_groups.insert(0, self.EXP_OPERATORS)
                            break

                    elif (isinstance(lhs, (int, float))
                          and middle in ['**', '^']):
                        # Split the float into two operations (1.2 => 1 x 0.2)
                        rhs, new_part = rh_op.op_value.split('.')
                        rh_op = Operand(self._to_int(rhs))

                        # For genuine floats, we need to restart the for-loop
                        if new_part:
                            new_part = self._to_float('0.' + str(new_part))
                            group.insert(i + offset, new_part)
                            group.insert(i + offset, '*')

                            group.insert(i + offset, rh_op)
                            group.insert(i + offset, middle)
                            group.insert(i + offset, lh_op)
                            offset += 2
                            remaining_op_groups.insert(0, self.EXP_OPERATORS)
                            break

                    elif isinstance(lhs, (int, float, BinaryOp)):
                        group.insert(i + offset, self._to_float(rhs))
                        group.insert(i + offset, '*')
                        group.insert(i + offset, lh_op)
                        # Re-run the EXP_OPERATORS loop
                        remaining_op_groups.insert(0, self.EXP_OPERATORS)
                        break

                # Convert any remaining strings which look like floats,
                # into floats.
                if self.is_float_like(lh_op):
                    lh_op = Operand(self._to_float(lh_op))
                if self.is_float_like(rh_op):
                    rh_op = Operand(self._to_float(rh_op))

                # Turn lhs, middle and rhs into actual BinaryOp instances
                # Note: Power operations are only allowed if the lhs is not a
                # BinaryOp.
                if is_exp and not isinstance(lh_op.op_value, BinaryOp):
                    res = Power(lh_op, rh_op)
                elif c[1].lower() in self.DIV_OPERATORS:
                    res = Division(lh_op, rh_op)
                else:
                    res = Multiplication(lh_op, rh_op)

                group.insert(i + offset, res)
                # Account for the 3 items removed (lhs, middle, rhs) and
                # the one item added (res).
                offset -= 2

        # If the group has whittled down to a single BinaryOp then just return
        # the Op itself.
        if len(group) == 1:
            return group[0]

        return group


if __name__ == "__main__":
    import doctest
    doctest.testmod()
