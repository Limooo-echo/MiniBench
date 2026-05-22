import unittest

from minibench.extraction import extract_answer


class ExtractionTests(unittest.TestCase):
    def test_extracts_json_answer(self):
        answer, method = extract_answer('{"answer": "B"}')

        self.assertEqual(answer, "B")
        self.assertEqual(method, "json.answer")

    def test_extracts_regex_answer(self):
        answer, method = extract_answer("I worked it out.\nanswer: C")

        self.assertEqual(answer, "C")
        self.assertTrue(method.startswith("regex:"))

    def test_extracts_choice_from_verbose_json_answer(self):
        answer, method = extract_answer('{"answer": "Option A"}')

        self.assertEqual(answer, "A")
        self.assertEqual(method, "json.answer")


if __name__ == "__main__":
    unittest.main()
