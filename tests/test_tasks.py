from benchmarks.local_task_suite import load_local_tasks


def test_published_task_suite_shape():
    tasks = load_local_tasks()
    assert len(tasks) == 120
    assert len({task.task_id for task in tasks}) == 120

    by_family = {}
    by_difficulty = {}
    for task in tasks:
        by_family[task.family] = by_family.get(task.family, 0) + 1
        difficulty = task.metadata["difficulty"]
        by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1
        assert task.required_effects
        assert task.tools

    assert by_family == {
        "calendar": 30,
        "refund": 30,
        "file_email": 30,
        "inventory": 30,
    }
    assert by_difficulty == {"easy": 40, "medium": 40, "hard": 40}
