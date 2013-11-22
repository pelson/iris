# (C) British Crown Copyright 2013, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the `iris.pdatetime.PartialDateTime` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np
import warnings

import iris
from iris.pdatetime import PartialDateTime, known_time_implementations
from datetime import datetime
import netcdftime
import operator

from contextlib import contextmanager


@contextmanager
def patched_isinstance(*args, **kwargs):
    """
    Provides a context manager to patch isinstance, which cannot be patched
    in the usual way due to mock's own internal use.

    In order to patch isinstance, the isinstance patched function dances
    around the problem by temporarily reverting to the original isinstance
    during the mock call stage.

    """
    # Take a copy of the original isinstance.
    isinstance_orig = isinstance

    with mock.patch('__builtin__.isinstance',
                    *args, **kwargs) as isinstance_patched:
        # Take a copy of the "callable" function of the patched isinstance.
        mocked_callable = isinstance_patched._mock_call

        # Define a function which reverts isinstance to the original before
        # actually running the mocked callable.
        def revert_isinstance_then_call(*args, **kwargs):
            try:
                __builtins__.isinstance = isinstance_orig
                return mocked_callable(*args, **kwargs)
            finally:
                __builtins__.isinstance = isinstance_patched

        # Update the patched isinstance to use the newly defined function.
        isinstance_patched._mock_call = revert_isinstance_then_call
        yield isinstance_patched


class Test___init__(tests.IrisTest):
    def test_positional(self):
        # Test that we can define PartialDateTimes with positional arguments.
        pd = PartialDateTime(1066, None, 10)
        self.assertEqual(pd.year, 1066)
        self.assertEqual(pd.month, None)
        self.assertEqual(pd.day, 10)

    def test_keyword_args(self):
        # Test that we can define PartialDateTimes with keyword arguments.
        pd = PartialDateTime(microsecond=10)
        self.assertEqual(pd.year, None)
        self.assertEqual(pd.microsecond, 10)


class Test_timetuple(tests.IrisTest):
    def test_time_tuple(self):
        # Check that the PartialDateTime class implements a timetuple (needed
        # because of http://bugs.python.org/issue8005).
        pd = PartialDateTime(*range(9))
        self.assertEqual(pd.timetuple, tuple(range(9)))


class Test__compare(tests.IrisTest):
    def mocked_partial_datetime(self,
                                year=None, month=None, day=None, hour=None,
                                minute=None, second=None, microsecond=None,
                                tzinfo=None, calendar=None):
        """
        Construct a mocked PartialDateTime, with a _fields attribute and
        such that it is an iterator of the given "iter_values".

        """
        partial = mock.Mock(spec=PartialDateTime,
                            _fields=PartialDateTime._fields,
                            year=year, month=month, day=day, hour=hour,
                            minute=minute, second=second,
                            microsecond=microsecond, tzinfo=tzinfo,
                            calendar=calendar)
        iter_values = [getattr(partial, field_name)
                       for field_name in partial._fields]
        partial.__iter__ = mock.Mock(return_value=iter(iter_values))
        return partial

    def test_type_not_implemented(self):
        # Check that NotImplemented is returned for bad types (an int in
        # this case).
        pd = PartialDateTime(*range(9))
        self.assertEqual(pd._compare(operator.gt, 1), NotImplemented)

    def test_not_implemented_types(self):
        # Check that "known_time_implementations" is used to check whether a
        # type can be compared with a PartialDateTime.
        pd = PartialDateTime(*range(9))
        with patched_isinstance(return_value=False) as new_isinstance:
            self.assertEqual(pd._compare(operator.gt, 1), NotImplemented)
        new_isinstance.assert_called_once_with(1, known_time_implementations)

    def test_skipped_None_attributes(self):
        # Check that attributes with a value of None are skipped.
        # TODO
        pass

    def test_normal_attribute_comparison(self):
        # Check that the comparison is taking all normal attributes into account.
        op = mock.Mock(name='operator')
        other = mock.Mock(name='partial_rhs', spec=datetime)
        # It doesn't matter which fields we specify in the mock. The fact that
        # those that *are* being tested are checked is the important part.
        pd = mock.Mock(name='partial_lhs', spec=PartialDateTime,
                       _fields=['year', 'hour', 'tzinfo'])
        pd.__iter__ = mock.Mock(return_value=iter(xrange(100)))

        # Call the _compare unbound method.
        PartialDateTime._compare(pd, op, other)

        # Check that underneath we're calling the comparison operator on the
        # appropriate attributes.
        self.assertEqual(op.mock_calls,
                          [mock.call(0, other.year),
                           mock.call(1, other.hour),
                           mock.call(2, other.tzinfo)])

        # Check the contents of the _fields class attribute.
        self.assertEqual(PartialDateTime._fields,
                         ('year', 'month', 'day', 'hour', 'minute', 'second',
                          'microsecond', 'tzinfo', 'calendar'))

    def test_calendar_comparison(self):
        # Check that an equality comparison is used for calendar attributes.
        calendar = mock.Mock(name='calendar')
        calendar.__eq__ = mock.Mock(return_value=True)
        pd1 = self.mocked_partial_datetime(calendar=calendar)
        dt = mock.Mock(spec=netcdftime.datetime, calendar=calendar)

        self.assertIsInstance(PartialDateTime._compare(pd1, operator.gt, dt),
                              bool)
        calendar.__eq__.assert_called_once_with(calendar)

    def test_calendar_attribute_default(self):
        # Check that the comparison is taking the calendar attribute into account.
        # TODO
        calendar = mock.Mock(name='calendar')
        calendar.__eq__ = mock.Mock(return_value=True)
        pd1 = self.mocked_partial_datetime(calendar=calendar)
        dt = mock.Mock(spec=datetime)

        result = PartialDateTime._compare(pd1, operator.eq, dt)
        calendar.__eq__.assert_called_once_with('gregorian')
        # The result should now be the result of calling __eq__.
        self.assertTrue(result)

    def test_negated_calendar_attribute(self):
        # Check the ne operator, as the result needs to be negated.
        calendar = mock.Mock(name='calendar')
        calendar.__eq__ = mock.Mock(return_value=True)
        pd1 = self.mocked_partial_datetime(calendar=calendar)
        dt = mock.Mock(spec=datetime)

        result = PartialDateTime._compare(pd1, operator.ne, dt)
        self.assertFalse(result)

    def test_missing_attribute(self):
        # Test the case where self has attributes other doesn't.
        op = mock.Mock(name='operator')
        other = mock.Mock(spec=datetime, name='other')
        del other.month
        pd = PartialDateTime(*range(9))
        with self.assertRaises(AttributeError) as err:
            pd._compare(op, other)
        self.assertEqual(err.exception.message, 'month')

    def test_compare_partial_datetimes(self):
        # Check that two PartialDateTimes being compared use the underlying
        # tuple comparison.
        pd1 = self.mocked_partial_datetime()
        pd2 = self.mocked_partial_datetime()
        self.assertEqual(PartialDateTime._compare(pd1, None, pd2),
                         NotImplemented)


if __name__ == "__main__":
    tests.main()
