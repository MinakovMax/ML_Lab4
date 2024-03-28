from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    """
    Класс для вычисления длины шага.

    Parameters
    ----------
    lambda_ : float, optional
        Начальная скорость обучения. По умолчанию 1e-3.
    s0 : float, optional
        Параметр для вычисления скорости обучения. По умолчанию 1.
    p : float, optional
        Степенной параметр для вычисления скорости обучения. По умолчанию 0.5.
    iteration : int, optional
        Текущая итерация. По умолчанию 0.

    Methods
    -------
    __call__()
        Вычисляет скорость обучения на текущей итерации.
    """
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Вычисляет скорость обучения по формуле lambda * (s0 / (s0 + t))^p.

        Returns
        -------
        float
            Скорость обучения на текущем шаге.
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    """
    Перечисление для выбора функции потерь.

    Attributes
    ----------
    MSE : auto
        Среднеквадратическая ошибка.
    MAE : auto
        Средняя абсолютная ошибка.
    LogCosh : auto
        Логарифм гиперболического косинуса от ошибки.
    Huber : auto
        Функция потерь Хьюбера.
    """
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    Базовый класс для всех методов градиентного спуска.

    Parameters
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float, optional
        Параметр скорости обучения. По умолчанию 1e-3.
    loss_function : LossFunction, optional
        Функция потерь, которая будет оптимизироваться. По умолчанию MSE.

    Attributes
    ----------
    w : np.ndarray
        Вектор весов модели.
    lr : LearningRate
        Скорость обучения.
    loss_function : LossFunction
        Функция потерь.

    Methods
    -------
    step(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Шаг градиентного спуска.
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов на основе градиента. Метод шаблон.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по весам. Метод шаблон.
    calc_loss(x: np.ndarray, y: np.ndarray) -> float
        Вычисление значения функции потерь.
    predict(x: np.ndarray) -> np.ndarray
        Вычисление прогнозов на основе признаков x.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE, isBasis: bool = False):
        """
        Инициализация базового класса для градиентного спуска.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Функция потерь, которая будет оптимизирована.

        Attributes
        ----------
        w : np.ndarray
            Начальный вектор весов, инициализированный случайным образом.
        lr : LearningRate
            Экземпляр класса для вычисления скорости обучения.
        loss_function : LossFunction
            Выбранная функция потерь.
        """
        if isBasis:
            self.w: np.ndarray = np.random.rand(dimension + 1)
        else: 
            self.w: np.ndarray = np.random.rand(dimension) 

        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function
        self.isBasis: bool = isBasis
            

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Выполнение одного шага градиентного спуска.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Разность между текущими и обновленными весами.
        """

        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Шаблон функции для обновления весов. Должен быть переопределен в подклассах.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь по весам.

        Returns
        -------
        np.ndarray
            Разность между текущими и обновленными весами. Этот метод должен быть реализован в подклассах.
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь по весам, поддерживает MSE и LogCosh.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам.
        """
        # Для MSE: градиент = 2/n * X^T * (Xw - y)
            
        if self.loss_function == LossFunction.MSE: 
            predictions = x @ self.w
            error = predictions - y
            gradient = 2 / len(y) * x.T @ error
            return gradient
        elif self.loss_function == LossFunction.LogCosh:
            # Для LogCosh: градиент = 1/n * X^T * tanh(Xw - y)
            predictions = x @ self.w
            error = predictions - y
            gradient = 1 / len(y) * x.T @ np.tanh(error)
            return gradient
        else:
            raise ValueError("Unsupported loss function")

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Вычисление значения функции потерь с использованием текущих весов.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        float
            Значение функции потерь.
        """
        predictions = self.predict(x)  # Получаем предсказания
        error = predictions - y
        

        if self.loss_function == LossFunction.MSE:
            # Для MSE: среднеквадратичная ошибка
            loss = np.mean((error) ** 2)
        elif self.loss_function == LossFunction.LogCosh:
            # Для LogCosh: log(cosh(predictions - y))
            loss = np.mean(np.log(np.cosh(error)))
        else:
            raise ValueError("Unsupported loss function")
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Расчет прогнозов на основе признаков x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        np.ndarray
            Прогнозируемые значения.
        """
        # Пример векторизованной реализации предсказаний (может потребоваться заменить на конкретную модель)
        predictions = np.dot(x, self.w)  # Предсказания y_pred = x * w
        return predictions


class VanillaGradientDescent(BaseDescent):
    """
    Класс полного градиентного спуска.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с учетом градиента.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по весам.
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов на основе градиента.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь по весам.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        new = self.lr.lambda_ * gradient
        self.w -= new
        
        # Возврат разности весов
        return self.w

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь по весам.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам.
        """
        return super().calc_gradient(x, y)


class StochasticDescent(VanillaGradientDescent):
    """
    Класс стохастического градиентного спуска.

    Parameters
    ----------
    batch_size : int, optional
        Размер мини-пакета. По умолчанию 50.

    Attributes
    ----------
    batch_size : int
        Размер мини-пакета.

    Методы
    -------
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по мини-пакетам.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50, min_loss = 0.29,
                 loss_function: LossFunction = LossFunction.MSE, isBasis: bool = False):

        super().__init__(dimension, lambda_, loss_function, isBasis)
        self.batch_size = batch_size
        
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь по мини-пакетам.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам, вычисленный по мини-пакету.
        """
        
        # Генерируем случайные индексы для мини-пакета размером batch_size
        idx = np.random.choice(len(x), size=self.batch_size, replace=False)
        
        x_batch = x[idx]  # Выбираем подмножество признаков из x с использованием сгенерированных индексов
        y_batch = y[idx]  # Выбираем подмножество целевых переменных из y с использованием сгенерированных индексов
                    # Для MSE: Вычисляем градиент по формуле для MSE
        predictions = x_batch.dot(self.w)
        errors = predictions - y_batch
        
        if self.loss_function == LossFunction.MSE:
            gradient = 2/x_batch.shape[0] * x_batch.T.dot(errors)
        elif self.loss_function == LossFunction.LogCosh:            
            gradient = np.dot(x_batch.T, np.tanh(errors)) / len(x_batch)
        else:
            raise NotImplementedError("This loss function is not supported yet.")    
        
        return gradient

 


class MomentumDescent(VanillaGradientDescent):
    """
    Класс градиентного спуска с моментом.

    Параметры
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float
        Параметр скорости обучения.
    loss_function : LossFunction
        Оптимизируемая функция потерь.

    Атрибуты
    ----------
    alpha : float
        Коэффициент момента.
    h : np.ndarray
        Вектор момента для весов.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с использованием момента.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, alpha: float = 0.9, loss_function: LossFunction = LossFunction.MSE, 
                 isBasis: bool = False):
        """
        Инициализация класса градиентного спуска с моментом.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Оптимизируемая функция потерь.
        """
        super().__init__(dimension, lambda_, loss_function, isBasis)
        self.alpha = alpha

        if (isBasis):
            self.h = np.zeros(dimension + 1)
        else:
            self.h = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием момента.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """

        # Обновление вектора момента
        self.h = self.alpha * self.h + self.lr.lambda_ * gradient
        
        # Обновление весов
        weight_difference = -self.h
        self.w += weight_difference
        
        # Возврат изменения весов
        return weight_difference


class Adam(VanillaGradientDescent):
    """
    Класс градиентного спуска с адаптивной оценкой моментов (Adam).

    Параметры
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float
        Параметр скорости обучения.
    loss_function : LossFunction
        Оптимизируемая функция потерь.

    Атрибуты
    ----------
    eps : float
        Малая добавка для предотвращения деления на ноль.
    m : np.ndarray
        Векторы первого момента.
    v : np.ndarray
        Векторы второго момента.
    beta_1 : float
        Коэффициент распада для первого момента.
    beta_2 : float
        Коэффициент распада для второго момента.
    iteration : int
        Счетчик итераций.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с использованием адаптивной оценки моментов.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE, isBasis: bool = False):
        """
        Инициализация класса Adam.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Оптимизируемая функция потерь.
        """
        super().__init__(dimension, lambda_, loss_function, isBasis)
        self.eps: float = 1e-8

        if isBasis:
            self.m: np.ndarray = np.zeros(dimension + 1)
            self.v: np.ndarray = np.zeros(dimension + 1)
        else:
            self.m: np.ndarray = np.zeros(dimension)
            self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием адаптивной оценки моментов.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        
        # Увеличиваем счетчик итераций на 1 при каждом вызове метода update_weights.
        self.iteration += 1

        # Обновляем вектор первого момента m с учетом текущего градиента и коэффициента beta_1.
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient

        # Обновляем вектор второго момента v с учетом текущего градиента и коэффициента beta_2.
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * gradient**2

        # Вычисляем скорректированный первый момент m_hat для учета bias изначальной инициализации m.
        m_hat = self.m / (1 - self.beta_1**self.iteration)

        # Вычисляем скорректированный второй момент v_hat для учета bias изначальной инициализации v.
        v_hat = self.v / (1 - self.beta_2**self.iteration)

        # Обновляем веса с использованием адаптивной оценки моментов по алгоритму Adam.
        update = self.lr.lambda_ * m_hat / (np.sqrt(v_hat) + self.eps)       

        # Обновляем веса
        self.w -= update

        # Возвращаем обновленные веса
        return self.w  

class BaseDescentReg(BaseDescent):
    """
    Базовый класс для градиентного спуска с регуляризацией.

    Параметры
    ----------
    *args : tuple
        Аргументы, передаваемые в базовый класс.
    mu : float, optional
        Коэффициент регуляризации. По умолчанию равен 0.
    **kwargs : dict
        Ключевые аргументы, передаваемые в базовый класс.

    Атрибуты
    ----------
    mu : float
        Коэффициент регуляризации.

    Методы
    -------
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь с учетом L2 регуляризации по весам.
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        Инициализация базового класса для градиентного спуска с регуляризацией.
        """
        super().__init__(*args, **kwargs)           
        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь и L2 регуляризации по весам.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь с учетом L2 регуляризации по весам.
        """
        # Вычисляем градиент основной функции потерь без учёта регуляризации
        gradient_without_reg = super().calc_gradient(x, y)
        
        # Создаем копию вектора весов, где смещение (первый элемент) занулен
        w_reg = np.copy(self.w)
        
        if self.isBasis:
            w_reg[0] = 0  # Исключаем смещение из регуляризации
            
        # Рассчитываем градиент регуляризации, применяя коэффициент регуляризации к зануленному вектору весов
        l2_gradient = self.mu * w_reg
        
        # Вычисляем итоговый градиент, комбинируя градиент основной функции потерь и градиент регуляризации
        total_gradient = gradient_without_reg + l2_gradient
        
        return total_gradient


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Класс полного градиентного спуска с регуляризацией.
    
    Этот класс комбинирует подход полного градиентного спуска с механизмом L2 регуляризации для
    оптимизации функции потерь. Он наследует основную логику обновления весов из VanillaGradientDescent
    и добавляет регуляризацию из BaseDescentReg.
    """

    def __init__(self, *args, **kwargs):
        """
        Инициализирует VanillaGradientDescentReg с заданными параметрами.
        """
        super().__init__(*args, **kwargs)

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        
               
        # Вычисляем градиент основной функции потерь без учета регуляризации
        #predictions = x.dot(self.w)
        #error = predictions - y
        #basic_gradient = (2/n) * x.T.dot(error)
        
        # Вычисляем базовый градиент с учетом функции потерь
        basic_gradient = super(VanillaGradientDescent, self).calc_gradient(x, y)

        # Создаем копию вектора весов, где смещение (первый элемент) занулен
        w_reg = np.copy(self.w)
        
        if self.isBasis:
            w_reg[0] = 0
        
        # Рассчитываем градиент регуляризации
        l2_gradient = self.mu * w_reg
        
        # Вычисляем итоговый градиент, комбинируя базовый градиент и градиент регуляризации
        total_gradient = basic_gradient + l2_gradient
        
        return total_gradient


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Класс стохастического градиентного спуска с регуляризацией.
    
    Объединяет механизм стохастического градиентного спуска с L2 регуляризацией
    для оптимизации функции потерь. Этот класс реализует подход стохастического
    градиентного спуска, позволяя более эффективно обрабатывать большие объемы данных
    за счет выполнения обновления весов на основе каждого отдельного наблюдения или небольшого батча.
    При этом применяется регуляризация для предотвращения переобучения модели.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисляет градиент функции потерь для стохастического градиентного спуска с учетом L2 регуляризации.
        """
        # Явный вызов calc_gradient из StochasticDescent
        basic_gradient = StochasticDescent.calc_gradient(self, x, y)

        # Применяем L2 регуляризацию, исключая влияние на смещение (bias)
        w_reg = np.copy(self.w)
        
        if self.isBasis:
            w_reg[0] = 0  # Исключаем смещение из регуляризации

        reg_gradient = self.mu * w_reg
        
        return basic_gradient + reg_gradient
    



class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Класс градиентного спуска с моментом и регуляризацией.
    
    Объединяет механизм градиентного спуска с моментом с L2 регуляризацией
    для оптимизации функции потерь. Этот класс использует концепцию момента для
    ускорения сходимости в процессе оптимизации и применяет регуляризацию для
    контроля сложности модели и предотвращения переобучения.
    """
    
    def __init__(self, *args, **kwargs):
        # Инициализируем параметры суперклассов
        super().__init__(*args, **kwargs)
        
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисляет градиент функции потерь для градиентного спуска с моментом и регуляризацией.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь с учетом момента и L2 регуляризации по весам.
        """
        # Вычисляем основной градиент с учетом момента из MomentumDescent
        basic_gradient = MomentumDescent.calc_gradient(self, x, y)

        # Добавляем регуляризацию, исключая влияние на смещение
        w_reg = np.copy(self.w)
        
        if self.isBasis:
            w_reg[0] = 0  # Исключаем смещение из регуляризации
        
        reg_gradient = self.mu * w_reg
        
        # Объединяем градиенты
        total_gradient = basic_gradient + reg_gradient
        
        return total_gradient
    



class AdamReg(BaseDescentReg, Adam):
    """
    Класс адаптивного градиентного алгоритма с регуляризацией (AdamReg).
    
    Объединяет алгоритм оптимизации Adam с L2 регуляризацией для более эффективной
    оптимизации функции потерь и контроля за сложностью модели. Подходит для работы
    как с малыми, так и с большими объемами данных и обладает адаптивной скоростью обучения
    для каждого параметра модели.
    """
    
    def __init__(self, *args, **kwargs):
        # Инициализируем параметры суперклассов
        super().__init__(*args, **kwargs)

        
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисляет градиент функции потерь для алгоритма Adam с учетом L2 регуляризации.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь с учетом адаптивного механизма Adam и L2 регуляризации по весам.
        """
        # Вычисляем градиент с использованием алгоритма Adam
        basic_gradient = Adam.calc_gradient(self, x, y)

        # Применяем L2 регуляризацию, исключая влияние на смещение
        w_reg = np.copy(self.w)
        w_reg[0] = 0  # Исключаем смещение из регуляризации
        reg_gradient = self.mu * w_reg
        
        # Комбинируем градиенты
        total_gradient = basic_gradient + reg_gradient
        return total_gradient


def get_descent(descent_config: dict) -> BaseDescent:
    """
    Создает экземпляр класса градиентного спуска на основе предоставленной конфигурации.

    Параметры
    ----------
    descent_config : dict
        Словарь конфигурации для выбора и настройки класса градиентного спуска. Должен содержать ключи:
        - 'descent_name': строка, название метода спуска ('full', 'stochastic', 'momentum', 'adam').
        - 'regularized': булево значение, указывает на необходимость использования регуляризации.
        - 'kwargs': словарь дополнительных аргументов, передаваемых в конструктор класса спуска.

    Возвращает
    -------
    BaseDescent
        Экземпляр класса, реализующего выбранный метод градиентного спуска.

    Исключения
    ----------
    ValueError
        Вызывается, если указано неправильное имя метода спуска.

    Примеры
    --------
    >>> descent_config = {
    ...     'descent_name': 'full',
    ...     'regularized': True,
    ...     'kwargs': {'dimension': 10, 'lambda_': 0.01, 'mu': 0.1, 'isBasis': False }
    ... }
    >>> descent = get_descent(descent_config)
    >>> isinstance(descent, BaseDescent)
    True
    """
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
