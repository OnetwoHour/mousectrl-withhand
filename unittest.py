import unittest
from unittest import mock
import mousecontrol


m = mock.Mock(side_effect=Exception())
m.side_effect = ['Up', 'Down', 'Right', 'Left', 'None']


class Test(unittest.TestCase):
    def test_result(self, var):
        print(var)
        test = var in ['Up', 'Down', 'Right', 'Left', 'None']
        self.assertEqual(test, True)


if __name__ == '__main__':
    t = Test()

    var = mousecontrol.main()
    m.assert_not_called()

    for i in range(10):
        t.test_result(m())
        t.test_result(next(var))

    m.assert_called()
