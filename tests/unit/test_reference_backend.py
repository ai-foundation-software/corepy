import pytest

from corepy.backend.reference import ReferenceBackend


class TestReferenceBackend:
    def test_add_scalars(self):
        assert ReferenceBackend.add(1, 2) == 3
        assert ReferenceBackend.add(1.5, 2.5) == 4.0

    def test_add_lists(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        assert ReferenceBackend.add(a, b) == [5, 7, 9]

    def test_add_broadcasting(self):
        # Specific to add implementation
        a = [1, 2]
        b = 10
        assert ReferenceBackend.add(a, b) == [11, 12]
        assert ReferenceBackend.add(b, a) == [11, 12]

    def test_sub_scalars(self):
        assert ReferenceBackend.sub(5, 3) == 2

    def test_sub_lists(self):
        a = [5, 5]
        b = [3, 2]
        assert ReferenceBackend.sub(a, b) == [2, 3]

    def test_mul_scalars(self):
        assert ReferenceBackend.mul(3, 4) == 12

    def test_mul_lists(self):
        a = [2, 3]
        b = [4, 5]
        assert ReferenceBackend.mul(a, b) == [8, 15]

    def test_div_scalars(self):
        assert ReferenceBackend.div(10, 2) == 5.0

    def test_div_lists(self):
        a = [10, 20]
        b = [2, 4]
        assert ReferenceBackend.div(a, b) == [5.0, 5.0]

    def test_matmul(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[1.0, 0.0], [0.0, 1.0]]  # Identity
        result = ReferenceBackend.matmul(a, b)
        assert result == a

        a = [[1, 2], [3, 4]]
        b = [[2, 0], [1, 2]]
        # 1*2+2*1 = 4
        # 1*0+2*2 = 4
        # 3*2+4*1 = 10
        # 3*0+4*2 = 8
        expected = [[4, 4], [10, 8]]
        assert ReferenceBackend.matmul(a, b) == expected

    def test_matmul_mismatch(self):
        a = [[1, 2]]  # 1x2
        b = [[1, 2], [3, 4], [5, 6]]  # 3x2. 2 != 3
        with pytest.raises(AssertionError):
            ReferenceBackend.matmul(a, b)
