"""The extension keeps the API of the reference dump (tests/api_dump.py, dev/overhaul-2.0.md T1.1).

A name, an argument or a writable property of the reference that the extension no longer has fails
the test, unless `api_dump.ACCEPTED` lists the difference with its reason (CONTRIBUTING.md: the
internals may change, with the reason "internal"; the user-facing API keeps a deprecated alias).
"""

import os
import unittest

from tests import api_dump


@unittest.skipUnless(os.path.isfile(api_dump.DEFAULT_OUTPUT), "no reference dump")
class TestApiDump(unittest.TestCase):
    def test_the_reference_api_is_kept(self):
        reference = api_dump.load(api_dump.DEFAULT_OUTPUT)
        self.assertEqual(api_dump.incompatibilities(reference, api_dump.dump()), [])

    def test_every_accepted_difference_occurs(self):
        """An entry of ACCEPTED that no longer occurs (the name is back, or the text of the
        difference changed) is stale."""
        reference = api_dump.load(api_dump.DEFAULT_OUTPUT)
        problems = set(api_dump.compare(reference, api_dump.dump()))
        self.assertEqual(sorted(set(api_dump.ACCEPTED) - problems), [])


if __name__ == "__main__":
    unittest.main()
