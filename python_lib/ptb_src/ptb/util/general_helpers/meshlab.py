import pymeshlab


class PyMeshLabHelper:
    @staticmethod
    def remeshing(edge_len, in_file, out_file, iterations : int = 10):
        """

        Args:
            edge_len: Sets the target length for the remeshed mesh edges.
            in_file: mesh to wrap
            out_file: save name
            iterations: Number of iterations of the remeshing operations to repeat on the mesh.

        Returns:

        """
        ms = pymeshlab.MeshSet()
        try:
            ms.load_new_mesh(in_file)
            val = pymeshlab.PureValue(edge_len)
            ms.remeshing_isotropic_explicit_remeshing(targetlen=val, iterations=iterations)
            ms.save_current_mesh(out_file)
        except pymeshlab.PyMeshLabException:
            return

    @staticmethod
    def wrap(alpha_fraction, offset_fraction, in_file, out_file):
        """

        Args:
            alpha_fraction: the size of the ball (fraction):
            offset_fraction: Offset added to the surface (fraction):
            in_file: mesh to wrap
            out_file: save name

        Returns:

        """
        ms = pymeshlab.MeshSet()
        try:
            ms.load_new_mesh(in_file)
            alpha_fraction_val = pymeshlab.PureValue(alpha_fraction)
            offset_fraction_val = pymeshlab.PureValue(offset_fraction)
            ms.generate_alpha_wrap(alpha_fraction=alpha_fraction_val, offset_fraction=offset_fraction_val)
            ms.save_current_mesh(out_file)
        except pymeshlab.PyMeshLabException:
            return

