import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from test_server import _content_matches_goal, _validate_task_relevance


def test_topic_validation_accepts_russian_inflection():
    assert _content_matches_goal('Циклы в Python', 'Напишите функцию с использованием цикла for')


def test_advanced_task_not_rejected_for_empty_starter_skeleton():
    valid, notes = _validate_task_relevance(
        'Классы в Python',
        'Реализуйте класс BankAccount с методами пополнения и снятия средств.',
        'class BankAccount:\n    pass\n',
        'advanced',
    )
    assert valid


def test_lexical_mismatch_is_warning_not_blocking_failure():
    valid, notes = _validate_task_relevance(
        'Обработка исключений',
        'Реализуйте безопасную функцию чтения числа и обработайте ошибочный ввод.',
        'def read_number(value):\n    pass\n',
        'intermediate',
    )
    assert valid
    assert notes


def test_strip_function_bodies_removes_ready_made_class_methods():
    from test_server import _strip_function_bodies
    source = (
        "class Calculator:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n\n"
        "def helper(x):\n"
        "    return x * 2\n"
    )
    skeleton = _strip_function_bodies(source)
    assert "return a + b" not in skeleton
    assert "return x * 2" not in skeleton
    assert skeleton.count("# TODO: реализуй функцию") == 2
    assert skeleton.count("pass") == 2
