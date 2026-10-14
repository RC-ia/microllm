import unittest

from corrigir_dataset import clean_dataset, strip_markdown


class MarkdownCleanupTests(unittest.TestCase):
    def test_removes_markers_but_keeps_words_and_list_items(self):
        source = "# Título\n\n**Texto** com *ênfase*.\n- primeiro\n2. segundo\n> citação"
        self.assertEqual(strip_markdown(source), "Título Texto com ênfase. primeiro segundo citação")

    def test_preserves_code_and_cleans_fences_inline_code_and_links(self):
        source = "Veja `x = 1` e [documentação](https://exemplo.test).\n\n```python\nprint('oi')\n```"
        result = strip_markdown(source)
        self.assertIn("Veja x = 1 e documentação (https://exemplo.test).", result)
        self.assertIn("print('oi')", result)
        self.assertNotIn("```", result)

    def test_table_becomes_readable_and_ruler_disappears(self):
        source = "| Nome | Valor |\n| --- | --- |\n| a | b |"
        result = strip_markdown(source)
        self.assertEqual(result, "Nome — Valor a — b")

    def test_flattens_newlines_and_removes_response_backslashes(self):
        result = strip_markdown("linha 1\\n\\nlinha 2\nlinha 3\\text{valor}")
        self.assertEqual(result, "linha 1 linha 2 linha 3text{valor}")
        self.assertNotIn("\\", result)
        self.assertNotIn("\n", result)

    def test_repairs_missing_space_after_sentence_punctuation(self):
        self.assertEqual(strip_markdown("Olá!Como você?Tudo certo?"),
                         "Olá! Como você? Tudo certo?")

    def test_dataset_structure_and_non_content_fields_are_preserved(self):
        data = [{"messages": [{"role": "assistant", "content": "**Olá**", "reasoning_content": "**keep**"}]}]
        cleaned, changed = clean_dataset(data)
        self.assertEqual(changed, 1)
        self.assertEqual(cleaned[0]["messages"][0]["content"], "Olá")
        self.assertEqual(cleaned[0]["messages"][0]["reasoning_content"], "**keep**")


if __name__ == "__main__":
    unittest.main()


