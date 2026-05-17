import unittest

from minibench.extraction import extract_answer


class ExtractionTests(unittest.TestCase):
    def test_extracts_json_answer(self):
        answer, method = extract_answer('{"answer": "391"}')

        self.assertEqual(answer, "391")
        self.assertEqual(method, "json.answer")

    def test_extracts_regex_answer(self):
        answer, method = extract_answer("I worked it out.\nanswer: alpha-42")

        self.assertEqual(answer, "alpha-42")
        self.assertTrue(method.startswith("regex:"))


if __name__ == "__main__":
    unittest.main()
