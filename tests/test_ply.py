"""``utils.ply``: getting a face count out of a header, and refusing otherwise.

The ``coarsen`` stage sizes its face target as a fraction of its input mesh's
face count, so this is the one place the pipeline reads a mesh file at all.
What has to hold is that it reads the *header* -- both binary byte orders
included, since a real ``reconstruct_mesh.ply`` is binary -- and that
everything it cannot answer comes back as one refusal a caller can act on
rather than as a plausible wrong number.
"""
import tempfile
import unittest
from pathlib import Path

from pgs_recon.utils.ply import MAX_HEADER_BYTES, NotAPlyHeader, face_count

ASCII_HEADER = """ply
format ascii 1.0
comment made by a test
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
"""


class PlyCase(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def write(self, name: str, text: str, tail: bytes = b'') -> Path:
        path = self.tmp / name
        path.write_bytes(text.encode() + tail)
        return path


class TestReadsTheCount(PlyCase):

    def test_ascii(self):
        self.assertEqual(2, face_count(self.write('a.ply', ASCII_HEADER)))

    def test_binary_little_endian(self):
        # The real case: OpenMVS writes binary, and the header is still text.
        header = ASCII_HEADER.replace('format ascii 1.0',
                                      'format binary_little_endian 1.0')
        # Followed by bytes that are not valid UTF-8, which is why the reader
        # opens in binary mode and compares byte strings.
        path = self.write('b.ply', header, tail=bytes(range(256)))
        self.assertEqual(2, face_count(path))

    def test_binary_big_endian(self):
        header = ASCII_HEADER.replace('format ascii 1.0',
                                      'format binary_big_endian 1.0')
        self.assertEqual(2, face_count(self.write('c.ply', header)))

    def test_crlf_line_endings(self):
        crlf = ASCII_HEADER.replace('\n', '\r\n')
        self.assertEqual(2, face_count(self.write('d.ply', crlf)))

    def test_the_vertex_count_is_not_mistaken_for_it(self):
        # Both blocks say `element <name> <count>`, and vertex comes first.
        self.assertEqual(2, face_count(self.write('e.ply', ASCII_HEADER)))

    def test_a_face_element_before_the_vertex_one(self):
        # Element order is the writer's choice; nothing requires vertex first.
        swapped = """ply
format ascii 1.0
element face 7
property list uchar int vertex_indices
element vertex 4
property float x
end_header
"""
        self.assertEqual(7, face_count(self.write('f.ply', swapped)))

    def test_a_real_face_count(self):
        header = ASCII_HEADER.replace('element face 2', 'element face 26003288')
        self.assertEqual(26003288, face_count(self.write('g.ply', header)))

    def test_a_mesh_with_no_faces_is_zero_not_a_refusal(self):
        # A point cloud written as a PLY declares `element face 0`. That is an
        # answer, and a caller taking a ratio of it gets a target of one face
        # -- which is what makes the floor in `coarsen_target` load-bearing.
        header = ASCII_HEADER.replace('element face 2', 'element face 0')
        self.assertEqual(0, face_count(self.write('h.ply', header)))


class TestRefusals(PlyCase):
    """Everything it cannot answer, as one exception type.

    A caller's recovery is the same in every case -- state the target outright,
    or drop the stage -- so these are one class rather than a taxonomy.
    """

    def test_not_a_ply_at_all(self):
        path = self.write('a.ply', 'fake reconstruct_mesh.ply\n')
        with self.assertRaises(NotAPlyHeader) as ctx:
            face_count(path)
        self.assertIn(str(path), str(ctx.exception))

    def test_an_obj(self):
        with self.assertRaises(NotAPlyHeader):
            face_count(self.write('a.obj', 'v 0 0 0\nf 1 1 1\n'))

    def test_an_empty_file(self):
        with self.assertRaises(NotAPlyHeader):
            face_count(self.write('a.ply', ''))

    def test_a_header_that_never_ends(self):
        # A truncated write, or a file that merely starts with the magic line.
        with self.assertRaises(NotAPlyHeader):
            face_count(self.write('a.ply', 'ply\nformat ascii 1.0\n'))

    def test_a_header_with_no_face_element(self):
        cloud = 'ply\nformat ascii 1.0\nelement vertex 4\nend_header\n'
        with self.assertRaises(NotAPlyHeader) as ctx:
            face_count(self.write('a.ply', cloud))
        self.assertIn('no face element', str(ctx.exception))

    def test_an_unreadable_count(self):
        header = ASCII_HEADER.replace('element face 2', 'element face lots')
        with self.assertRaises(NotAPlyHeader):
            face_count(self.write('a.ply', header))

    def test_a_file_whose_header_runs_past_the_cap(self):
        # The cap is what keeps a mis-bound artifact -- a multi-gigabyte mesh
        # under some other format -- from being read into memory in full.
        comments = 'comment ' + 'x' * 200 + '\n'
        n = MAX_HEADER_BYTES // len(comments) + 2
        with self.assertRaises(NotAPlyHeader) as ctx:
            face_count(self.write('a.ply',
                                  'ply\nformat ascii 1.0\n' + comments * n
                                  + 'element face 2\nend_header\n'))
        self.assertIn('end_header', str(ctx.exception))

    def test_a_missing_file_raises_oserror_not_this(self):
        # Callers catch both; keeping them distinct means "cannot open" does
        # not arrive dressed as "not a PLY".
        with self.assertRaises(OSError):
            face_count(self.tmp / 'nope.ply')


if __name__ == '__main__':
    unittest.main()
