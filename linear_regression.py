from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent

from tqdm.notebook import tqdm

class LinearRegression:
    """
    Класс линейной регрессии.

    Parameters
    ----------
    descent_config : dict
        Конфигурация градиентного спуска.
    tolerance : float, optional
        Критерий остановки для квадрата евклидова нормы разности весов. По умолчанию равен 1e-4.
    max_iter : int, optional
        Критерий остановки по количеству итераций. По умолчанию равен 300.

    Attributes
    ----------
    descent : BaseDescent
        Экземпляр класса, реализующего градиентный спуск.
    tolerance : float
        Критерий остановки для квадрата евклидова нормы разности весов.
    max_iter : int
        Критерий остановки по количеству итераций.
    loss_history : List[float]
        История значений функции потерь на каждой итерации.

    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300, min_loss: float = 0):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)
        self.tolerance: float = tolerance
        self.max_iter: int = max_iter
        self.min_loss = min_loss

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Обучение модели линейной регрессии, подбор весов для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        self : LinearRegression
            Возвращает экземпляр класса с обученными весами.

        """
        # Добавляем столбец из единиц к x для учета смещения
        if self.descent.isBasis:
            ones = np.ones((x.shape[0], 1))
            x = np.hstack([ones, x])  # Модифицированный x с добавленным столбцом для смещения
        

        iter_count = 1
        
        loss = self.descent.calc_loss(x, y)
        self.loss_history.append(loss)
        
        while iter_count < self.max_iter:
            
            gradient = self.descent.calc_gradient(x, y)
            weight_diff = self.descent.update_weights(gradient)
            
            loss = self.descent.calc_loss(x, y)            
            self.loss_history.append(loss)
            print(loss)
            if loss > 1000:
                print("Ошибка зашкаливает явно идем в неверном направлении")
                break

            if loss < self.min_loss:
                print('Достигнута минимальная установленная ошибка {}'.format(self.min_loss))
                break


            if np.isnan(self.descent.w).any():
                print("Веса содержат значения NaN. Обучение остановлено.")
                break

            
            if np.linalg.norm(weight_diff) < self.tolerance:
                print("Разница в весах меньше заданного уровня толерантности. Обучение остановлено.")
                break
            
            iter_count += 1   
        
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Прогнозирование целевых переменных для набора данных x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        prediction : np.ndarray
            Массив прогнозируемых значений.
        """
        ones = np.ones((x.shape[0], 1))
        x_bias = np.hstack([ones, x])  # Также добавляем столбец из единиц к x при предсказании
        return self.descent.predict(x_bias)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Расчёт значения функции потерь для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        loss : float
            Значение функции потерь.
        """
        ones = np.ones((x.shape[0], 1))
        x_bias = np.hstack([ones, x])  # Также добавляем столбец из единиц к x при предсказании
        return self.descent.calc_loss(x_bias, y)
