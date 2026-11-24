"""Точка входа в текущий запуск экспериментов PINN.

Команда ``python -m experiments`` и установленная команда ``experiments``
запускают то же приложение Hydra, что и ``python -m experiments.pinn``.
"""

from experiments.pinn.__main__ import main

if __name__ == "__main__":
    main()
