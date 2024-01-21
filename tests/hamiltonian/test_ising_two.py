from ising.hamiltonian import parametrized_ising_two


class TestIsing:
    def test_ising_side_1(self):
        ham = parametrized_ising_two(side=1, h=1)

        assert ham is not None
        assert ham.sparse_repr.paulis == ["X"]

    def test_ising_side_2(self):
        ham = parametrized_ising_two(side=2, h=1)

        paulis = [
            "XIII",
            "IXII",
            "IIXI",
            "IIIX",
            "ZZII",
            "ZIZI",
            "IZIZ",
            "IIZZ",
        ]

        assert ham is not None
        assert set(ham.sparse_repr.paulis.to_labels()) == set(paulis)
