#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>


/// @brief Структура для представления двумерного вектора.
struct Vector
{
    float x, y; // Компоненты вектора.

    // Конструктор по умолчанию.
    Vector() : x(0.0f), y(0.0f) {}

    // Конструктор с параметрами.
    Vector(float x, float y) : x(x), y(y) {}

    // Преобразование в структуру sf::Vector2f.
    operator sf::Vector2f() const {
        return sf::Vector2f(x, y);
    }

    /// @brief Сложение с другим вектором.
    /// @param vector Вектор, который будет добавлен к данному вектору.
    /// @return Ссылка на измененный вектор.
    Vector& add(const Vector& vector) {
        x += vector.x;
        y += vector.y;

        return *this;
    }

    /// @brief Вычитание другого вектора.
    /// @param vector Вектор, который будет вычтен из данного вектора.
    /// @return Ссылка на измененный вектор.
    Vector& sub(const Vector& vector) {
        x -= vector.x;
        y -= vector.y;

        return *this;
    }

    /// @brief Умножение на скаляр.
    /// @param num Скаляр, на который будет умножен вектор.
    /// @return Ссылка на измененный вектор.
    Vector& mult(float num) {
        x *= num;
        y *= num;

        return *this;
    }

    /// @brief Деление на скаляр.
    /// @param num Скаляр, на который будет разделен вектор.
    /// @return Ссылка на измененный вектор.
    Vector& div(float num) {
        if (num != 0.0f) {
            x /= num;
            y /= num;
        }

        return *this;
    }

    /// @brief Вычисление скалярного произведения с другим вектором.
    /// @param vector Другой вектор.
    /// @return Скалярное произведение.
    float dot(const Vector& vector) const {
        return x * vector.x + y * vector.y;
    }

    /// @brief Вычисление псевдо-скалярного произведения с другим вектором.
    /// @param vector Другой вектор.
    /// @return Псевдо-скалярное произведение.
    float pseudoDot(const Vector& vector) const {
        return x * vector.y - y * vector.x;
    }

    /// @brief Нормализация вектора.
    /// @return Ссылка на измененный вектор.
    Vector& normalize() {
        float norm = std::sqrt(x * x + y * y);

        if (norm != 0.0f) {
            x /= norm;
            y /= norm;
        }

        return *this;
    }

    /// @brief Вычисление нормы вектора.
    /// @return Норма вектора.
    float norm() const {
        return std::sqrt(x * x + y * y);
    }
};


/// @brief Функция сложения двух векторов.
/// @param vector1 Первый вектор.
/// @param vector2 Второй вектор.
/// @return Результат сложения - новый вектор.
Vector vector_add(const Vector& vector1, const Vector& vector2) {
    float x = vector1.x + vector2.x;
    float y = vector1.y + vector2.y;

    return Vector(x, y);
}


/// @brief Функция вычитания векторов: vector1 - vector2.
/// @param vector1 Петрвый вектор.
/// @param vector2 Второй вектор.
/// @return Результат сложения - новый вектор.
Vector vector_sub(const Vector& vector1, const Vector& vector2) {
    float x = vector1.x - vector2.x;
    float y = vector1.y - vector2.y;

    return Vector(x, y);
}


/// @brief Функция умножения вектора на скаляр.
/// @param vector Вектор, который нужно умножить.
/// @param num Скаляр, на который нужно умножить вектор.
/// @return Результат умножения - новый вектор.
Vector vector_mult(const Vector& vector, float num) {
    float x = vector.x * num;
    float y = vector.y * num;

    return Vector(x, y);
}


/// @brief Функция деления вектора на скаляр.
/// @param vector Вектор, который нужно разделить.
/// @param num Скаляр, на который нужно разделить вектор.
/// @return Результат деления - новый вектор.
Vector vector_div(const Vector& vector, float num) {
    float x = vector.x;
    float y = vector.y;

    if (num != 0.0f) {
        x /= num;
        y /= num;
    }

    return Vector(x, y);
}


/// @brief Функция для получения вектора нормали.
/// @param vector Вектор, для которого нужно получить нормаль.
/// @return Нормальный вектор.
Vector vector_get_normal_vector(const Vector& vector) {
    float x = vector.y;
    float y = -1.0f * vector.x;

    return Vector(x, y);
}


/// @brief Функция вычисления скалярного произведения двух векторов.
/// @param vector1 Первый вектор.
/// @param vector2 Второй вектор.
/// @return  Результат скалярного произведения.
float vector_dot(const Vector& vector1, const Vector& vector2) {

    return vector1.x * vector2.x + vector1.y * vector2.y;
}


/// @brief Функция вычисления нормы вектора.
/// @param vector Вектор, для которого нужно вычислить норму.
/// @return  Норма вектора.
float vector_norm(const Vector& vector) {
    return std::sqrt(vector.x * vector.x + vector.y * vector.y);
}


/// @brief Функция нормализации вектора.
/// @param vector vector Вектор, который нужно нормализовать.
/// @return  Нормализованный вектор.
Vector vector_normalize(const Vector& vector) {
    Vector out;
    float norm = std::sqrt(vector.x * vector.x + vector.y * vector.y);

    if (norm != 0.0f) {
        float x = vector.x / norm;
        float y = vector.y / norm;

        out = Vector(x, y);
    }

    return out;
}


// .******************************* Константы *******************************.
const float fps = 300.0f;
const float dt = 1.0f / fps;
const Vector gravity(0.0f, 9.8f * 5);


/// @brief Функция проверки пересечения проекций отрезков по осям "x" и "y" (одновременно).
/// @param A Точка начала отрезка A.
/// @param B Точка конца отрезка A.
/// @param C Точка начала отрезка B.
/// @param D Точка конца отрезка B.
/// @return  true, если проекции отрезков пересекаются, в противном случае - false.
bool projectionsIntersection(
    const Vector& A, 
    const Vector& B, 
    const Vector& C, 
    const Vector& D) {
    
    float x_11, x_12, x_21, x_22, y_11, y_12, y_21, y_22;
    float x1_max, y1_max, x2_min, y2_min;
    bool x_intersection = false;
    bool y_intersection = false;

    // Определяем левую и правую точки проекции первой прямой по оси "x".
    if (A.x <= B.x) {
        x_11 = A.x;
        x_12 = B.x;
    } else {
        x_11 = B.x;
        x_12 = A.x;
    }
    // Определяем левую и правую точки проекции второй прямой по оси "x".
    if (C.x <= D.x) {
        x_21 = C.x;
        x_22 = D.x;
    } else {
        x_21 = D.x;
        x_22 = C.x;
    }
    // Определяем верхнюю и нижнюю точки проекции первой прямой по оси "y".
    if (A.y <= B.y) {
        y_11 = A.y;
        y_12 = B.y;
    } else {
        y_11 = B.y;
        y_12 = A.y;
    }
    // Определяем верхнюю и нижнюю точки проекции второй прямой по оси "y".
    if (C.y <= D.y) {
        y_21 = C.y;
        y_22 = D.y;
    } else {
        y_21 = D.y;
        y_22 = C.y;
    }

    // Определяем правую точку среди левых.
    x1_max = std::max(x_11, x_21);
    // Определяем левую точку среди правых.
    x2_min = std::min(x_12, x_22);
    // Определяем верхнюю точку среди нижних.
    y1_max = std::max(y_11, y_21);
    // Определяем нижнюю точку среди верхних.
    y2_min = std::min(y_12, y_22);

    // Условие пересечения проекций по оси "x": 
    // правая точка среди левых должна быть левее левой точки среди правых.
    if(x1_max <= x2_min) {
        x_intersection = true;
    }
    // Условие пересечения проекций по оси "y": 
    // верхняя точка среди нижних должна быть ниже нижней точки среди верхних.
    if(y1_max <= y2_min) {
        y_intersection = true;
    }

    // Необходимо, чтобы одновременно пересекались обе проекции.
    return x_intersection * y_intersection;

}


/// @brief Функция проверки пересечение двух отрезков.
/// @param A Начальная точка первого отрезка.
/// @param B Конечная точка первого отрезка.
/// @param C Начальная точка второго отрезка.
/// @param D Конечная точка второго отрезка.
/// @return true, если пересекаются, иначе false.
bool segmentsIntersection(
    const Vector& A, 
    const Vector& B, 
    const Vector& C, 
    const Vector& D) {

    bool intersection = true;

    // Необходиоме условие: пересечение проекций прямых по осям "x" и "y" одновременно.
    if(!projectionsIntersection(A, B, C, D)) return !intersection;

    Vector DA = vector_sub(A, D);
    Vector DB = vector_sub(B, D);
    Vector DC = vector_sub(C, D);
    float DAxDC = DC.pseudoDot(DA);
    float DBxDC = DC.pseudoDot(DB);

    // Достаточное условие: точки одного отрезка должны лежать по разные стороны от второго отрезка.
    // Т.е. углы поворота должны быть противоположных знаков.
    // (знак псевдо-скалярного произведения совпадает со знаком угла поворота).
    if(DAxDC * DBxDC > 0.0f) return !intersection;

    Vector AB = vector_sub(B, A);
    Vector AC = vector_sub(C, A);
    Vector AD = vector_sub(D, A);
    float ADxAB = AB.pseudoDot(AD);
    float ACxAB = AB.pseudoDot(AC);

    // Достаточное условие: точки второго отрезка тоже должны лежать по разные стороны от первого отрезка.
    if(ADxAB * ACxAB > 0.0f) return !intersection;

    return intersection;
}


// Функция определения минимального расстояния от точки "E" до отрезка "AB".
// На входе имеем вектора всех точек для отрезка "AB" и точки "E".
// Возвращаем вектор "EF", где "E" - исходная точка, а "F" - ближайшая точка на отрезке "AB" к точке "E".

/// @brief Функция определения минимального расстояния от точки "E" до отрезка "AB".
/// @param A Первая точка отрезка "AB".
/// @param B Вторая точка отрезка "AB".
/// @param E Точка Е.
/// @return Вектор "EF", где "E" - исходная точка, а "F" - ближайшая точка на отрезке "AB" к точке "E".
Vector minDistance(
    const Vector& A, 
    const Vector& B, 
    const Vector& E) {

    Vector AB = vector_sub(B, A);
    Vector BE = vector_sub(E, B);
    Vector AE = vector_sub(E, A);

    // Определяем проекции соответствующих векторов.
    float ABoBE = AB.dot(BE);
    float ABoAE = AB.dot(AE);

    // Если данные вектора соноправлены, то к точке "E" ближайшей будет точка "B" (край отрезка).
    if(ABoBE >= 0.0f && ABoAE > 0.0f) return vector_sub(B, E);

    // Если данные вектора соноправлены, то к точке "E" ближайшей будет точка "A" (край отрезка).
    if(ABoBE < 0.0f && ABoAE <= 0.0f) return vector_sub(A, E);

    // Если данные вектора противоположно направлены, то к точке "E" ближайшей будет точка "F" (на отрезке).
    // EF = Проекция(AE на AB) - AE. Проекция(AE на AB) = ( (AE * AB) / |AB| )e_ab.
    if(ABoBE < 0.0f && ABoAE > 0.0f) {
        Vector e_ab = vector_normalize(AB);
        float projection = ABoAE / vector_norm(AB);

        return vector_sub(e_ab.mult(projection), AE);
    }

    // В случае ошибки.
    return Vector(0, 0);

}


/// @brief Структура, представляющая границу прямоугольной области для многоугольника.
struct Boundary
{
    float x_max, y_max, x_min, y_min; 
    Vector center;

    /// @brief Конструктор, инициализирующий границу на основе переданных вершин многоугольника.
    /// @param vertices Вектор вершин многоугольника.
    Boundary(const std::vector<Vector>& vertices) {
        x_min = std::numeric_limits<float>::max();
        y_min = std::numeric_limits<float>::max();
        x_max = std::numeric_limits<float>::min();
        y_max = std::numeric_limits<float>::min();

        // Определение максимальных и минимальных точек по осям среди вершин многоугольника.
        for (const Vector& vertex : vertices) {
            if (vertex.x < x_min) x_min = vertex.x;
            if (vertex.y < y_min) y_min = vertex.y;
            if (vertex.x > x_max) x_max = vertex.x;
            if (vertex.y > y_max) y_max = vertex.y;
        }

        // Определение центра прямоугольника (границы).
        float x = (x_max + x_min) / 2;
        float y = (y_max + y_min) / 2;
        center = Vector(x, y);
    }

    /// @brief Функция, определяющая попадает ли точка внутрь границы.
    /// @param position Координаты точки.
    /// @return true, если точка содержится внутри границы, иначе false.
    bool isContain(const Vector& position) const {
        if(
            position.x >= x_min && 
            position.x <= x_max && 
            position.y >= y_min && 
            position.y <= y_max) {

            return true;
        }

        return false;
    }

};


/// @brief Структура, представляющая собой многоугольник.
struct Polygon
{
    // Хранит вершины и границу.
    std::vector<Vector> vertices;
    Boundary boundary;
    Vector viscidity;

    /// @brief Конструктор, инициализирующий многоугольник.
    /// @param points Вектор вершин многоугольника.
    /// @param viscidity Вязкость многоугольника.
    Polygon(std::vector<Vector> points, Vector viscidity) : 
        vertices(points), 
        boundary(points), 
        viscidity(viscidity) 
    {}

    /// @brief Метод отрисовки многоугольника.
    /// @param window Окно для отрисовки.
    void draw(sf::RenderWindow& window) const {
        sf::ConvexShape polygon;
        polygon.setPointCount(vertices.size());

        // Задаем координаты вершин многоугольника
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            polygon.setPoint(i, vertices[i]);
        }

        // Устанавливаем цвет многоугольника
        polygon.setFillColor(sf::Color(211, 211, 211)); // Light Gray

        // Отрисовываем многоугольник
        window.draw(polygon);
    }

};


/// @brief Структура массовой частицы тела.
struct Particle
{
    float mass, radius;
    Vector position, velocity, force;
    sf::Color color;

    /// @brief Конструктор частицы.
    /// @param x_0 Координата x начального положения.
    /// @param y_0 Координата y начального положения.
    /// @param R Радиус частицы.
    /// @param M Масса частицы.
    /// @param color color Цвет частицы.
    Particle(float x_0, float y_0, float R, float M, sf::Color color) : 
        position(x_0, y_0), 
        velocity(0.0f, 0.0f), 
        force(0.0f, 0.0f), 
        mass(M),
        radius(R),
        color(color)
    {}

    /// @brief Прикладывает силу к частице.
    /// @param new_force Прикладываема сила.
    void applyForce(const Vector new_force) {
        force.add(new_force);
    }

    ///@brief Вычисление добавки к скорости и смещение частицы.
    void update() {
        force.add(vector_mult(gravity, mass));
        // v += Fdt / m
        velocity.add(force.mult(dt / mass));
        // r += vdt
        position.add(vector_mult(velocity, dt));
        force.mult(0.0f);
    }

    /// @brief Проверка на столкновение с другой частицей.
    /// @param particle Другая частица.
    /// @return true eсли произошло столкновение, false если нет.
    bool isCollideParticle(const Particle& particle) const {
        float distance = vector_sub(particle.position, position).norm();

        // Условие столкновения: distance < r1 + r2.
        if(distance < radius + particle.radius) {
            return true;
        }

        return false;
    }

    /// @brief Отражение частицы при столкновении с другой частицей.
    /// @param particle Другая частица.
    void collideParticle(Particle& particle) {
        // "Зона" пересечения: dist = r1 + r2 - |p2 - p1|.
        Vector normal = vector_sub(position, particle.position);
        float distance = radius + particle.radius - normal.norm();
        normal.normalize();
        Vector displacement = vector_mult(normal, 0.5f * distance);

        // "Раздвигаем" частицы.
        position.add(displacement);
        particle.position.sub(displacement);

        // Отражаем скорости: r = d - 2(d * n)n, 
        // где r - отраженный вектор, d - падающий вектор, n - нормаль.
        Vector this_reflection = vector_mult(normal, 2.0f * velocity.dot(normal));
        Vector other_reflection = vector_mult(normal, 2.0f * particle.velocity.dot(normal));

        velocity.sub(this_reflection);
        // Потеря энергии при столкновении.
        velocity.mult(0.75f);
        particle.velocity.sub(other_reflection);
        // Потеря энергии при столкновении.
        particle.velocity.mult(0.75f);
    }

    /// @brief Проверка на столкновение со "стенами".
    /// @param polygon Многоугольник, представляющий "стены".
    /// @return true если частица столкнулась со стеной, false если нет.
    bool isCollideWall(const Polygon& polygon) const {
        bool contain = true;

        // Проверяем попадает ли частица внутрь границы, содержащей многоугольник.
        if(!polygon.boundary.isContain(position)) return !contain;

        int count = 0;

        // Проверяем попадает ли частица внутрь многоугольника при помощи лучевого анализа (raycasting).
        // Берем луч, выравненный по оси "x" (от левого края экрана до нашей частицы), 
        // и проверяем количество пересечений этого луча с ребрами многоугольника.
        for (size_t i = 0; i < polygon.vertices.size(); ++i) {
            bool intersection = segmentsIntersection(
                position, 
                Vector(0.0f, position.y), 
                polygon.vertices[i], 
                polygon.vertices[(i + 1) % polygon.vertices.size()]);

            if(intersection) count ++;
        }

        // Условие попадания точки в многоугольник: пересечений нечетное количество.
        if(count % 2 != 0) return contain;

        return !contain;
    }

    /// @brief Отражение частицы при столкновении со "стеной".
    /// @param polygon Многоугольник, представляющий "стену".
    void collideWall(const Polygon& polygon) {
        float distance = std::numeric_limits<float>::max();
        Vector direction;

        // Определяем ближайшую точку многоугольника к нашей частице.
        // Находим вектор от частицы к этой точке.
        for (size_t i = 0; i < polygon.vertices.size(); ++i) {
            Vector A = polygon.vertices[i];
            Vector B = polygon.vertices[(i + 1) % polygon.vertices.size()];

            Vector ray = minDistance(A, B, position);
            float ray_norm = ray.norm();
            if(ray_norm < distance) {
                distance = ray_norm;
                direction = ray;
            }
        }

        direction.normalize();

        // Используем найденный вектор, как вектор отражения.
        position.add(vector_mult(direction, distance));

        // Отражаем скорость: r = d - 2(d * n)n.
        Vector reflection = vector_mult(direction, 2.0f * velocity.dot(direction));
        velocity.sub(reflection);

        // Потеря энергии при столкновении
        float velocity_projection = velocity.dot(polygon.viscidity);
        velocity = vector_mult(polygon.viscidity, 0.9f * velocity_projection);

    }

    /// @brief Отрисовка частицы.
    /// @param window Окно для отрисовки.
    void draw(sf::RenderWindow& window) const {
        sf::CircleShape circle(radius);
        circle.setFillColor(color);
        circle.setPosition(position.x - radius, position.y - radius);
        window.draw(circle);
    }

};


/// @brief Структура пружины, соединяющей две частицы.
struct Spring
{
    float stiffness, restLength, dampingFactor;
    sf::Color color;
    std::size_t head_index, tail_index;

    /// @brief Конструктор для объекта Spring.
    /// @param headIndex Индекс частицы, соединенной с началом пружины.
    /// @param tailIndex Индекс частицы, соединенной с концом пружины.
    /// @param restLength Длина пружины в нерастянутом состоянии.
    /// @param stiff Коэффициент жесткости пружины.
    /// @param damping Коэффициент демпфирования пружины.
    /// @param color Цвет пружины.
    Spring(
        std::size_t headIndex, 
        std::size_t tailIndex, 
        const float restLength, 
        const float stiff, 
        const float damping,
        sf::Color color) : 

        stiffness(stiff), 
        dampingFactor(damping), 
        restLength(restLength), 
        head_index(headIndex), 
        tail_index(tailIndex), 
        color(color)
    {}

    /// @brief Вычисляет силу, возникающую в пружине, и прикладывает ее к соединенным частицам.
    /// @param particles Вектор всех частиц.
    void update(std::vector<Particle>& particles) {
        // Определяем направление.
        Vector direction_head_tail = vector_sub(particles[tail_index].position, particles[head_index].position);
        // Определяем растяжение-сжатие.
        float extension = direction_head_tail.norm() - restLength;
        direction_head_tail.normalize();
        // Определяем относительную скорость.
        Vector relative_velocity = vector_sub(particles[tail_index].velocity, particles[head_index].velocity);

        // Упругая сила (закон Гука: F_упр = kx).
        float elastic_force = extension * stiffness;
        // Демпфирующая сила: F_демп = b(e_12 * v_отн).
        float damping_force = vector_dot(direction_head_tail, relative_velocity) * dampingFactor;

        // Сила пружины: F = (F_упр + F_демп)e.
        Vector force = direction_head_tail.mult(elastic_force + damping_force);

        // Прикладываем силы к частицам.
        particles[head_index].applyForce(force);
        particles[tail_index].applyForce(force.mult(-1.0f));
    }

    /// @brief Отрисовка пружины.
    ///  @param window Окно, в котором будет отрисована пружина.
    /// @param particles Вектор всех частиц.
    void draw(sf::RenderWindow& window, std::vector<Particle>& particles) const {
        sf::VertexArray line(sf::Lines, 2);
        line[0].position = particles[head_index].position;
        line[1].position = particles[tail_index].position;

        // Устанавливаем белый цвет линии
        line[0].color = color;
        line[1].color = color;

        window.draw(line);
    }

};


// O - tetramino:
// @ @
// @ @
/// @brief Создает тетрамино O и добавляет его частицы и пружины в соответствующие векторы.
/// @param particles Вектор частиц для добавления созданных частиц тетрамино.
/// @param springs Вектор пружин для добавления созданных пружин тетрамино.
/// @param colors Вектор цветов, из которых будет выбран цвет тетрамино.
/// @param x0 Координата x верхнего левого угла тетрамино.
/// @param y0 Координата y верхнего левого угла тетрамино.
/// @param radius Радиус частиц тетрамино.
/// @param mass Масса частиц тетрамино.
/// @param edge Длина ребра тетрамино.
/// @param diagonal Длина диагонали тетрамино.
/// @param stiff Коэффициент жесткости пружин тетрамино.
/// @param damp Коэффициент демпфирования пружин тетрамино.
void create_O_tetramino(
    std::vector<Particle>& particles, 
    std::vector<Spring>& springs, 
    const sf::Color color,
    float x0, float y0, 
    const float& radius, 
    const float& mass, 
    const float& edge, 
    const float& diagonal, 
    const float& stiff, 
    const float& damp) {

    std::size_t i = particles.size();

    particles.push_back(Particle(x0, y0, radius, mass, color)); // particle_0
    particles.push_back(Particle(x0 + 1.0f * edge, y0, radius, mass, color)); // particle_1
    particles.push_back(Particle(x0 + 2.0f * edge, y0, radius, mass, color)); // particle_2
    particles.push_back(Particle(x0, y0 + 1.0f * edge, radius, mass, color)); // particle_3
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_4
    particles.push_back(Particle(x0 + 2.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_5
    particles.push_back(Particle(x0, y0 + 2.0f * edge, radius, mass, color)); // particle_6
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_7
    particles.push_back(Particle(x0 + 2.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_8

    springs.push_back(Spring(i + 0, i + 1, edge, stiff, damp, color)); // spring_0_1
    springs.push_back(Spring(i + 0, i + 3, edge, stiff, damp, color)); // spring_0_3
    springs.push_back(Spring(i + 0, i + 4, diagonal, stiff, damp, color)); // spring_0_4
    springs.push_back(Spring(i + 1, i + 2, edge, stiff, damp, color)); // spring_1_2
    springs.push_back(Spring(i + 1, i + 3, diagonal, stiff, damp, color)); // spring_1_3
    springs.push_back(Spring(i + 1, i + 4, edge, stiff, damp, color)); // spring_1_4
    springs.push_back(Spring(i + 1, i + 5, diagonal, stiff, damp, color)); // spring_1_5
    springs.push_back(Spring(i + 2, i + 4, diagonal, stiff, damp, color)); // spring_2_4
    springs.push_back(Spring(i + 2, i + 5, edge, stiff, damp, color)); // spring_2_5
    springs.push_back(Spring(i + 3, i + 4, edge, stiff, damp, color)); // spring_3_4
    springs.push_back(Spring(i + 3, i + 6, edge, stiff, damp, color)); // spring_3_6
    springs.push_back(Spring(i + 3, i + 7, diagonal, stiff, damp, color)); // spring_3_7
    springs.push_back(Spring(i + 4, i + 5, edge, stiff, damp, color)); // spring_4_5
    springs.push_back(Spring(i + 4, i + 6, diagonal, stiff, damp, color)); // spring_4_6
    springs.push_back(Spring(i + 4, i + 7, edge, stiff, damp, color)); // spring_4_7
    springs.push_back(Spring(i + 4, i + 8, diagonal, stiff, damp, color)); // spring_4_8
    springs.push_back(Spring(i + 5, i + 7, diagonal, stiff, damp, color)); // spring_5_7
    springs.push_back(Spring(i + 5, i + 8, edge, stiff, damp, color)); // spring_5_8
    springs.push_back(Spring(i + 6, i + 7, edge, stiff, damp, color)); // spring_6_7
    springs.push_back(Spring(i + 7, i + 8, edge, stiff, damp, color)); // spring_7_8

}


// T - tetramino:
// @
// @ @
// @
/// @brief Создает тетрамино T и добавляет его частицы и пружины в соответствующие векторы.
/// @param particles Вектор частиц для добавления созданных частиц тетрамино.
/// @param springs Вектор пружин для добавления созданных пружин тетрамино.
/// @param colors Вектор цветов, из которых будет выбран цвет тетрамино.
/// @param x0 Координата x верхнего левого угла тетрамино.
/// @param y0 Координата y верхнего левого угла тетрамино.
/// @param radius Радиус частиц тетрамино.
/// @param mass Масса частиц тетрамино.
/// @param edge Длина ребра тетрамино.
/// @param diagonal Длина диагонали тетрамино.
/// @param stiff Коэффициент жесткости пружин тетрамино.
/// @param damp Коэффициент демпфирования пружин тетрамино.
void create_T_tetramino(
    std::vector<Particle>& particles, 
    std::vector<Spring>& springs, 
    const sf::Color color,
    float x0, float y0, 
    const float& radius, 
    const float& mass, 
    const float& edge, 
    const float& diagonal, 
    const float& stiff, 
    const float& damp) {

    std::size_t i = particles.size();

    particles.push_back(Particle(x0, y0, radius, mass, color)); // particle_0
    particles.push_back(Particle(x0 + 1.0f * edge, y0, radius, mass, color)); // particle_1
    particles.push_back(Particle(x0, y0 + 1.0f * edge, radius, mass, color)); // particle_2
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_3
    particles.push_back(Particle(x0 + 2.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_4
    particles.push_back(Particle(x0, y0 + 2.0f * edge, radius, mass, color)); // particle_5
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_6
    particles.push_back(Particle(x0 + 2.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_7
    particles.push_back(Particle(x0, y0 + 3.0f * edge, radius, mass, color)); // particle_8
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 3.0f * edge, radius, mass, color)); // particle_9

    springs.push_back(Spring(i + 0, i + 1, edge, stiff, damp, color)); // spring_0_1
    springs.push_back(Spring(i + 0, i + 2, edge, stiff, damp, color)); // spring_0_2
    springs.push_back(Spring(i + 0, i + 3, diagonal, stiff, damp, color)); // spring_0_3
    springs.push_back(Spring(i + 1, i + 2, diagonal, stiff, damp, color)); // spring_1_2
    springs.push_back(Spring(i + 1, i + 3, edge, stiff, damp, color)); // spring_1_3
    springs.push_back(Spring(i + 2, i + 3, edge, stiff, damp, color)); // spring_2_3
    springs.push_back(Spring(i + 2, i + 5, edge, stiff, damp, color)); // spring_2_5
    springs.push_back(Spring(i + 2, i + 6, diagonal, stiff, damp, color)); // spring_2_6
    springs.push_back(Spring(i + 3, i + 4, edge, stiff, damp, color)); // spring_3_4
    springs.push_back(Spring(i + 3, i + 5, diagonal, stiff, damp, color)); // spring_3_5
    springs.push_back(Spring(i + 3, i + 6, edge, stiff, damp, color)); // spring_3_6
    springs.push_back(Spring(i + 3, i + 7, diagonal, stiff, damp, color)); // spring_3_7
    springs.push_back(Spring(i + 4, i + 6, diagonal, stiff, damp, color)); // spring_4_6
    springs.push_back(Spring(i + 4, i + 7, edge, stiff, damp, color)); // spring_4_7
    springs.push_back(Spring(i + 5, i + 6, edge, stiff, damp, color)); // spring_5_6
    springs.push_back(Spring(i + 5, i + 8, edge, stiff, damp, color)); // spring_5_8
    springs.push_back(Spring(i + 5, i + 9, diagonal, stiff, damp, color)); // spring_5_9
    springs.push_back(Spring(i + 6, i + 7, edge, stiff, damp, color)); // spring_6_7
    springs.push_back(Spring(i + 6, i + 8, diagonal, stiff, damp, color)); // spring_6_8
    springs.push_back(Spring(i + 6, i + 9, edge, stiff, damp, color)); // spring_6_9
    springs.push_back(Spring(i + 8, i + 9, edge, stiff, damp, color)); // spring_8_9

}


// J1 - tetramino:
// @
// @ @ @
/// @brief Создает тетрамино J1 и добавляет его частицы и пружины в соответствующие векторы.
/// @param particles Вектор частиц для добавления созданных частиц тетрамино.
/// @param springs Вектор пружин для добавления созданных пружин тетрамино.
/// @param colors Вектор цветов, из которых будет выбран цвет тетрамино.
/// @param x0 Координата x верхнего левого угла тетрамино.
/// @param y0 Координата y верхнего левого угла тетрамино.
/// @param radius Радиус частиц тетрамино.
/// @param mass Масса частиц тетрамино.
/// @param edge Длина ребра тетрамино.
/// @param diagonal Длина диагонали тетрамино.
/// @param stiff Коэффициент жесткости пружин тетрамино.
/// @param damp Коэффициент демпфирования пружин тетрамино.
void create_J1_tetramino(
    std::vector<Particle>& particles, 
    std::vector<Spring>& springs, 
    const sf::Color color,
    float x0, float y0, 
    const float& radius, 
    const float& mass, 
    const float& edge, 
    const float& diagonal, 
    const float& stiff, 
    const float& damp) {

    std::size_t i = particles.size();

    particles.push_back(Particle(x0, y0, radius, mass, color)); // particle_0
    particles.push_back(Particle(x0 + 1.0f * edge, y0, radius, mass, color)); // particle_1
    particles.push_back(Particle(x0, y0 + 1.0f * edge, radius, mass, color)); // particle_2
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_3
    particles.push_back(Particle(x0 + 2.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_4
    particles.push_back(Particle(x0, y0 + 2.0f * edge, radius, mass, color)); // particle_5
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_6
    particles.push_back(Particle(x0 + 2.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_7
    particles.push_back(Particle(x0 + 3.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_8
    particles.push_back(Particle(x0 + 3.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_9

    springs.push_back(Spring(i + 0, i + 1, edge, stiff, damp, color)); // spring_0_1
    springs.push_back(Spring(i + 0, i + 2, edge, stiff, damp, color)); // spring_0_2
    springs.push_back(Spring(i + 0, i + 3, diagonal, stiff, damp, color)); // spring_0_3
    springs.push_back(Spring(i + 1, i + 2, diagonal, stiff, damp, color)); // spring_1_2
    springs.push_back(Spring(i + 1, i + 3, edge, stiff, damp, color)); // spring_1_3
    springs.push_back(Spring(i + 2, i + 3, edge, stiff, damp, color)); // spring_2_3
    springs.push_back(Spring(i + 2, i + 5, edge, stiff, damp, color)); // spring_2_5
    springs.push_back(Spring(i + 2, i + 6, diagonal, stiff, damp, color)); // spring_2_6
    springs.push_back(Spring(i + 3, i + 4, edge, stiff, damp, color)); // spring_3_4
    springs.push_back(Spring(i + 3, i + 5, diagonal, stiff, damp, color)); // spring_3_5
    springs.push_back(Spring(i + 3, i + 6, edge, stiff, damp, color)); // spring_3_6
    springs.push_back(Spring(i + 3, i + 7, diagonal, stiff, damp, color)); // spring_3_7
    springs.push_back(Spring(i + 4, i + 6, diagonal, stiff, damp, color)); // spring_4_6
    springs.push_back(Spring(i + 4, i + 7, edge, stiff, damp, color)); // spring_4_7
    springs.push_back(Spring(i + 4, i + 8, edge, stiff, damp, color)); // spring_4_8
    springs.push_back(Spring(i + 4, i + 9, diagonal, stiff, damp, color)); // spring_4_9
    springs.push_back(Spring(i + 5, i + 6, edge, stiff, damp, color)); // spring_5_6
    springs.push_back(Spring(i + 6, i + 7, edge, stiff, damp, color)); // spring_6_7
    springs.push_back(Spring(i + 7, i + 8, diagonal, stiff, damp, color)); // spring_7_8
    springs.push_back(Spring(i + 7, i + 9, edge, stiff, damp, color)); // spring_7_9
    springs.push_back(Spring(i + 8, i + 9, edge, stiff, damp, color)); // spring_8_9

}


// J2 - tetramino:
// @ @ @
//     @
/// @brief Создает тетрамино J2 и добавляет его частицы и пружины в соответствующие векторы.
/// @param particles Вектор частиц для добавления созданных частиц тетрамино.
/// @param springs Вектор пружин для добавления созданных пружин тетрамино.
/// @param colors Вектор цветов, из которых будет выбран цвет тетрамино.
/// @param x0 Координата x верхнего левого угла тетрамино.
/// @param y0 Координата y верхнего левого угла тетрамино.
/// @param radius Радиус частиц тетрамино.
/// @param mass Масса частиц тетрамино.
/// @param edge Длина ребра тетрамино.
/// @param diagonal Длина диагонали тетрамино.
/// @param stiff Коэффициент жесткости пружин тетрамино.
/// @param damp Коэффициент демпфирования пружин тетрамино.
void create_J2_tetramino(
    std::vector<Particle>& particles, 
    std::vector<Spring>& springs, 
    const sf::Color color,
    float x0, float y0, 
    const float& radius, 
    const float& mass, 
    const float& edge, 
    const float& diagonal, 
    const float& stiff, 
    const float& damp) {

    std::size_t i = particles.size();

    particles.push_back(Particle(x0, y0, radius, mass, color)); // particle_0
    particles.push_back(Particle(x0 + 1.0f * edge, y0, radius, mass, color)); // particle_1
    particles.push_back(Particle(x0 + 2.0f * edge, y0, radius, mass, color)); // particle_2
    particles.push_back(Particle(x0, y0 + 1.0f * edge, radius, mass, color)); // particle_3
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_4
    particles.push_back(Particle(x0 + 2.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_5
    particles.push_back(Particle(x0 + 3.0f * edge, y0, radius, mass, color)); // particle_6
    particles.push_back(Particle(x0 + 3.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_7
    particles.push_back(Particle(x0 + 2.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_8
    particles.push_back(Particle(x0 + 3.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_9

    springs.push_back(Spring(i + 0, i + 1, edge, stiff, damp, color)); // spring_0_1
    springs.push_back(Spring(i + 0, i + 3, edge, stiff, damp, color)); // spring_0_3
    springs.push_back(Spring(i + 0, i + 4, diagonal, stiff, damp, color)); // spring_0_4
    springs.push_back(Spring(i + 1, i + 2, edge, stiff, damp, color)); // spring_1_2
    springs.push_back(Spring(i + 1, i + 3, diagonal, stiff, damp, color)); // spring_1_3
    springs.push_back(Spring(i + 1, i + 4, edge, stiff, damp, color)); // spring_1_4
    springs.push_back(Spring(i + 1, i + 5, diagonal, stiff, damp, color)); // spring_1_5
    springs.push_back(Spring(i + 2, i + 4, diagonal, stiff, damp, color)); // spring_2_4
    springs.push_back(Spring(i + 2, i + 5, edge, stiff, damp, color)); // spring_2_5
    springs.push_back(Spring(i + 2, i + 6, edge, stiff, damp, color)); // spring_2_6
    springs.push_back(Spring(i + 2, i + 7, diagonal, stiff, damp, color)); // spring_2_7
    springs.push_back(Spring(i + 3, i + 4, edge, stiff, damp, color)); // spring_3_4
    springs.push_back(Spring(i + 4, i + 5, edge, stiff, damp, color)); // spring_4_5
    springs.push_back(Spring(i + 5, i + 7, edge, stiff, damp, color)); // spring_5_7
    springs.push_back(Spring(i + 5, i + 8, edge, stiff, damp, color)); // spring_5_8
    springs.push_back(Spring(i + 5, i + 9, diagonal, stiff, damp, color)); // spring_5_9
    springs.push_back(Spring(i + 6, i + 7, edge, stiff, damp, color)); // spring_6_7
    springs.push_back(Spring(i + 7, i + 8, diagonal, stiff, damp, color)); // spring_7_8
    springs.push_back(Spring(i + 7, i + 9, edge, stiff, damp, color)); // spring_7_9
    springs.push_back(Spring(i + 8, i + 9, edge, stiff, damp, color)); // spring_8_9

}


// I1 - tetramino:
// @
// @
// @
// @
/// @brief Создает тетрамино I1 и добавляет его частицы и пружины в соответствующие векторы.
/// @param particles Вектор частиц для добавления созданных частиц тетрамино.
/// @param springs Вектор пружин для добавления созданных пружин тетрамино.
/// @param colors Вектор цветов, из которых будет выбран цвет тетрамино.
/// @param x0 Координата x верхнего левого угла тетрамино.
/// @param y0 Координата y верхнего левого угла тетрамино.
/// @param radius Радиус частиц тетрамино.
/// @param mass Масса частиц тетрамино.
/// @param edge Длина ребра тетрамино.
/// @param diagonal Длина диагонали тетрамино.
/// @param stiff Коэффициент жесткости пружин тетрамино.
/// @param damp Коэффициент демпфирования пружин тетрамино.
void create_I1_tetramino(
    std::vector<Particle>& particles, 
    std::vector<Spring>& springs, 
    const sf::Color color,
    float x0, float y0, 
    const float& radius, 
    const float& mass, 
    const float& edge, 
    const float& diagonal, 
    const float& stiff, 
    const float& damp) {

    std::size_t i = particles.size();

    particles.push_back(Particle(x0, y0, radius, mass, color)); // particle_0
    particles.push_back(Particle(x0 + 1.0f * edge, y0, radius, mass, color)); // particle_1
    particles.push_back(Particle(x0, y0 + 1.0f * edge, radius, mass, color)); // particle_2
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_3
    particles.push_back(Particle(x0, y0 + 4.0f * edge, radius, mass, color)); // particle_4
    particles.push_back(Particle(x0, y0 + 2.0f * edge, radius, mass, color)); // particle_5
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_6
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 4.0f * edge, radius, mass, color)); // particle_7
    particles.push_back(Particle(x0, y0 + 3.0f * edge, radius, mass, color)); // particle_8
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 3.0f * edge, radius, mass, color)); // particle_9

    springs.push_back(Spring(i + 0, i + 1, edge, stiff, damp, color)); // spring_0_1
    springs.push_back(Spring(i + 0, i + 2, edge, stiff, damp, color)); // spring_0_2
    springs.push_back(Spring(i + 0, i + 3, diagonal, stiff, damp, color)); // spring_0_3
    springs.push_back(Spring(i + 1, i + 2, diagonal, stiff, damp, color)); // spring_1_2
    springs.push_back(Spring(i + 1, i + 3, edge, stiff, damp, color)); // spring_1_3
    springs.push_back(Spring(i + 2, i + 3, edge, stiff, damp, color)); // spring_2_3
    springs.push_back(Spring(i + 2, i + 5, edge, stiff, damp, color)); // spring_2_5
    springs.push_back(Spring(i + 2, i + 6, diagonal, stiff, damp, color)); // spring_2_6
    springs.push_back(Spring(i + 3, i + 5, diagonal, stiff, damp, color)); // spring_3_5
    springs.push_back(Spring(i + 3, i + 6, edge, stiff, damp, color)); // spring_3_6
    springs.push_back(Spring(i + 4, i + 7, edge, stiff, damp, color)); // spring_4_7
    springs.push_back(Spring(i + 4, i + 8, edge, stiff, damp, color)); // spring_4_8
    springs.push_back(Spring(i + 4, i + 9, diagonal, stiff, damp, color)); // spring_4_9
    springs.push_back(Spring(i + 5, i + 6, edge, stiff, damp, color)); // spring_5_6
    springs.push_back(Spring(i + 5, i + 8, edge, stiff, damp, color)); // spring_5_8
    springs.push_back(Spring(i + 5, i + 9, diagonal, stiff, damp, color)); // spring_5_9
    springs.push_back(Spring(i + 6, i + 8, diagonal, stiff, damp, color)); // spring_6_8
    springs.push_back(Spring(i + 6, i + 9, edge, stiff, damp, color)); // spring_6_9
    springs.push_back(Spring(i + 7, i + 8, diagonal, stiff, damp, color)); // spring_7_8
    springs.push_back(Spring(i + 7, i + 9, edge, stiff, damp, color)); // spring_7_9
    springs.push_back(Spring(i + 8, i + 9, edge, stiff, damp, color)); // spring_8_9

}


// I2 - tetramino:
// @ @ @ @
/// @brief Создает тетрамино I2 и добавляет его частицы и пружины в соответствующие векторы.
/// @param particles Вектор частиц для добавления созданных частиц тетрамино.
/// @param springs Вектор пружин для добавления созданных пружин тетрамино.
/// @param colors Вектор цветов, из которых будет выбран цвет тетрамино.
/// @param x0 Координата x верхнего левого угла тетрамино.
/// @param y0 Координата y верхнего левого угла тетрамино.
/// @param radius Радиус частиц тетрамино.
/// @param mass Масса частиц тетрамино.
/// @param edge Длина ребра тетрамино.
/// @param diagonal Длина диагонали тетрамино.
/// @param stiff Коэффициент жесткости пружин тетрамино.
/// @param damp Коэффициент демпфирования пружин тетрамино.
void create_I2_tetramino(
    std::vector<Particle>& particles, 
    std::vector<Spring>& springs, 
    const sf::Color color,
    float x0, float y0, 
    const float& radius, 
    const float& mass, 
    const float& edge, 
    const float& diagonal, 
    const float& stiff, 
    const float& damp) {

    std::size_t i = particles.size();

    particles.push_back(Particle(x0, y0, radius, mass, color)); // particle_0
    particles.push_back(Particle(x0 + 1.0f * edge, y0, radius, mass, color)); // particle_1
    particles.push_back(Particle(x0 + 2.0f * edge, y0, radius, mass, color)); // particle_2
    particles.push_back(Particle(x0, y0 + 1.0f * edge, radius, mass, color)); // particle_3
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_4
    particles.push_back(Particle(x0 + 2.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_5
    particles.push_back(Particle(x0 + 3.0f * edge, y0, radius, mass, color)); // particle_6
    particles.push_back(Particle(x0 + 3.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_7
    particles.push_back(Particle(x0 + 4.0f * edge, y0, radius, mass, color)); // particle_8
    particles.push_back(Particle(x0 + 4.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_9

    springs.push_back(Spring(i + 0, i + 1, edge, stiff, damp, color)); // spring_0_1
    springs.push_back(Spring(i + 0, i + 3, edge, stiff, damp, color)); // spring_0_3
    springs.push_back(Spring(i + 0, i + 4, diagonal, stiff, damp, color)); // spring_0_4
    springs.push_back(Spring(i + 1, i + 2, edge, stiff, damp, color)); // spring_1_2
    springs.push_back(Spring(i + 1, i + 3, diagonal, stiff, damp, color)); // spring_1_3
    springs.push_back(Spring(i + 1, i + 4, edge, stiff, damp, color)); // spring_1_4
    springs.push_back(Spring(i + 1, i + 5, diagonal, stiff, damp, color)); // spring_1_5
    springs.push_back(Spring(i + 2, i + 4, diagonal, stiff, damp, color)); // spring_2_4
    springs.push_back(Spring(i + 2, i + 5, edge, stiff, damp, color)); // spring_2_5
    springs.push_back(Spring(i + 2, i + 6, edge, stiff, damp, color)); // spring_2_6
    springs.push_back(Spring(i + 2, i + 7, diagonal, stiff, damp, color)); // spring_2_7
    springs.push_back(Spring(i + 3, i + 4, edge, stiff, damp, color)); // spring_3_4
    springs.push_back(Spring(i + 4, i + 5, edge, stiff, damp, color)); // spring_4_5
    springs.push_back(Spring(i + 5, i + 6, diagonal, stiff, damp, color)); // spring_5_6
    springs.push_back(Spring(i + 5, i + 7, edge, stiff, damp, color)); // spring_5_7
    springs.push_back(Spring(i + 6, i + 7, edge, stiff, damp, color)); // spring_6_7
    springs.push_back(Spring(i + 6, i + 8, edge, stiff, damp, color)); // spring_6_8
    springs.push_back(Spring(i + 6, i + 9, diagonal, stiff, damp, color)); // spring_6_9
    springs.push_back(Spring(i + 7, i + 8, diagonal, stiff, damp, color)); // spring_7_8
    springs.push_back(Spring(i + 7, i + 9, edge, stiff, damp, color)); // spring_7_9
    springs.push_back(Spring(i + 8, i + 9, edge, stiff, damp, color)); // spring_8_9

}


// Z1 - tetramino:
// @ @
//   @ @
/// @brief Создает тетрамино Z1 и добавляет его частицы и пружины в соответствующие векторы.
/// @param particles Вектор частиц для добавления созданных частиц тетрамино.
/// @param springs Вектор пружин для добавления созданных пружин тетрамино.
/// @param colors Вектор цветов, из которых будет выбран цвет тетрамино.
/// @param x0 Координата x верхнего левого угла тетрамино.
/// @param y0 Координата y верхнего левого угла тетрамино.
/// @param radius Радиус частиц тетрамино.
/// @param mass Масса частиц тетрамино.
/// @param edge Длина ребра тетрамино.
/// @param diagonal Длина диагонали тетрамино.
/// @param stiff Коэффициент жесткости пружин тетрамино.
/// @param damp Коэффициент демпфирования пружин тетрамино.
void create_Z1_tetramino(
    std::vector<Particle>& particles, 
    std::vector<Spring>& springs, 
    const sf::Color color,
    float x0, float y0, 
    const float& radius, 
    const float& mass, 
    const float& edge, 
    const float& diagonal, 
    const float& stiff, 
    const float& damp) {

    std::size_t i = particles.size();

    particles.push_back(Particle(x0, y0, radius, mass, color)); // particle_0
    particles.push_back(Particle(x0 + 1.0f * edge, y0, radius, mass, color)); // particle_1
    particles.push_back(Particle(x0 + 2.0f * edge, y0, radius, mass, color)); // particle_2
    particles.push_back(Particle(x0, y0 + 1.0f * edge, radius, mass, color)); // particle_3
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_4
    particles.push_back(Particle(x0 + 2.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_5
    particles.push_back(Particle(x0 + 3.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_6
    particles.push_back(Particle(x0 + 3.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_7
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_8
    particles.push_back(Particle(x0 + 2.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_9

    springs.push_back(Spring(i + 0, i + 1, edge, stiff, damp, color)); // spring_0_1
    springs.push_back(Spring(i + 0, i + 3, edge, stiff, damp, color)); // spring_0_3
    springs.push_back(Spring(i + 0, i + 4, diagonal, stiff, damp, color)); // spring_0_4
    springs.push_back(Spring(i + 1, i + 2, edge, stiff, damp, color)); // spring_1_2
    springs.push_back(Spring(i + 1, i + 3, diagonal, stiff, damp, color)); // spring_1_3
    springs.push_back(Spring(i + 1, i + 4, edge, stiff, damp, color)); // spring_1_4
    springs.push_back(Spring(i + 1, i + 5, diagonal, stiff, damp, color)); // spring_1_5
    springs.push_back(Spring(i + 2, i + 4, diagonal, stiff, damp, color)); // spring_2_4
    springs.push_back(Spring(i + 2, i + 5, edge, stiff, damp, color)); // spring_2_5
    springs.push_back(Spring(i + 3, i + 4, edge, stiff, damp, color)); // spring_3_4
    springs.push_back(Spring(i + 4, i + 5, edge, stiff, damp, color)); // spring_4_5
    springs.push_back(Spring(i + 4, i + 8, edge, stiff, damp, color)); // spring_4_8
    springs.push_back(Spring(i + 4, i + 9, diagonal, stiff, damp, color)); // spring_4_9
    springs.push_back(Spring(i + 5, i + 6, edge, stiff, damp, color)); // spring_5_6
    springs.push_back(Spring(i + 5, i + 7, diagonal, stiff, damp, color)); // spring_5_7
    springs.push_back(Spring(i + 5, i + 8, diagonal, stiff, damp, color)); // spring_5_8
    springs.push_back(Spring(i + 5, i + 9, edge, stiff, damp, color)); // spring_5_9
    springs.push_back(Spring(i + 6, i + 7, edge, stiff, damp, color)); // spring_6_7
    springs.push_back(Spring(i + 6, i + 9, diagonal, stiff, damp, color)); // spring_6_9
    springs.push_back(Spring(i + 7, i + 9, edge, stiff, damp, color)); // spring_7_9
    springs.push_back(Spring(i + 8, i + 9, edge, stiff, damp, color)); // spring_8_9

}


// Z2 - tetramino:
//   @
// @ @
// @
/// @brief Создает тетрамино Z2 и добавляет его частицы и пружины в соответствующие векторы.
/// @param particles Вектор частиц для добавления созданных частиц тетрамино.
/// @param springs Вектор пружин для добавления созданных пружин тетрамино.
/// @param colors Вектор цветов, из которых будет выбран цвет тетрамино.
/// @param x0 Координата x верхнего левого угла тетрамино.
/// @param y0 Координата y верхнего левого угла тетрамино.
/// @param radius Радиус частиц тетрамино.
/// @param mass Масса частиц тетрамино.
/// @param edge Длина ребра тетрамино.
/// @param diagonal Длина диагонали тетрамино.
/// @param stiff Коэффициент жесткости пружин тетрамино.
/// @param damp Коэффициент демпфирования пружин тетрамино.
void create_Z2_tetramino(
    std::vector<Particle>& particles, 
    std::vector<Spring>& springs, 
    const sf::Color color,
    float x0, float y0, 
    const float& radius, 
    const float& mass, 
    const float& edge, 
    const float& diagonal, 
    const float& stiff, 
    const float& damp) {

    std::size_t i = particles.size();

    particles.push_back(Particle(x0, y0, radius, mass, color)); // particle_0
    particles.push_back(Particle(x0 + 1.0f * edge, y0, radius, mass, color)); // particle_1
    particles.push_back(Particle(x0 - edge, y0 + 1.0f * edge, radius, mass, color)); // particle_2
    particles.push_back(Particle(x0, y0 + 1.0f * edge, radius, mass, color)); // particle_3
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_4
    particles.push_back(Particle(x0 - edge, y0 + 2.0f * edge, radius, mass, color)); // particle_5
    particles.push_back(Particle(x0, y0 + 2.0f * edge, radius, mass, color)); // particle_6
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_7
    particles.push_back(Particle(x0 - edge, y0 + 3.0f * edge, radius, mass, color)); // particle_8
    particles.push_back(Particle(x0, y0 + 3.0f * edge, radius, mass, color)); // particle_9

    springs.push_back(Spring(i + 0, i + 1, edge, stiff, damp, color)); // spring_0_1
    springs.push_back(Spring(i + 0, i + 3, edge, stiff, damp, color)); // spring_0_3
    springs.push_back(Spring(i + 0, i + 4, diagonal, stiff, damp, color)); // spring_0_4
    springs.push_back(Spring(i + 1, i + 3, diagonal, stiff, damp, color)); // spring_1_3
    springs.push_back(Spring(i + 1, i + 4, edge, stiff, damp, color)); // spring_1_4
    springs.push_back(Spring(i + 2, i + 3, edge, stiff, damp, color)); // spring_2_3
    springs.push_back(Spring(i + 2, i + 5, edge, stiff, damp, color)); // spring_2_5
    springs.push_back(Spring(i + 2, i + 6, diagonal, stiff, damp, color)); // spring_2_6
    springs.push_back(Spring(i + 3, i + 4, edge, stiff, damp, color)); // spring_3_4
    springs.push_back(Spring(i + 3, i + 5, diagonal, stiff, damp, color)); // spring_3_5
    springs.push_back(Spring(i + 3, i + 6, edge, stiff, damp, color)); // spring_3_6
    springs.push_back(Spring(i + 3, i + 7, diagonal, stiff, damp, color)); // spring_3_7
    springs.push_back(Spring(i + 4, i + 6, diagonal, stiff, damp, color)); // spring_4_6
    springs.push_back(Spring(i + 4, i + 7, edge, stiff, damp, color)); // spring_4_7
    springs.push_back(Spring(i + 5, i + 6, edge, stiff, damp, color)); // spring_5_6
    springs.push_back(Spring(i + 5, i + 8, edge, stiff, damp, color)); // spring_5_8
    springs.push_back(Spring(i + 5, i + 9, diagonal, stiff, damp, color)); // spring_5_9
    springs.push_back(Spring(i + 6, i + 7, edge, stiff, damp, color)); // spring_6_7
    springs.push_back(Spring(i + 6, i + 8, diagonal, stiff, damp, color)); // spring_6_8
    springs.push_back(Spring(i + 6, i + 9, edge, stiff, damp, color)); // spring_6_9
    springs.push_back(Spring(i + 8, i + 9, edge, stiff, damp, color)); // spring_8_9

}


// L - tetramino:
// @ @
//   @
//   @
/// @brief Создает тетрамино L и добавляет его частицы и пружины в соответствующие векторы.
/// @param particles Вектор частиц для добавления созданных частиц тетрамино.
/// @param springs Вектор пружин для добавления созданных пружин тетрамино.
/// @param colors Вектор цветов, из которых будет выбран цвет тетрамино.
/// @param x0 Координата x верхнего левого угла тетрамино.
/// @param y0 Координата y верхнего левого угла тетрамино.
/// @param radius Радиус частиц тетрамино.
/// @param mass Масса частиц тетрамино.
/// @param edge Длина ребра тетрамино.
/// @param diagonal Длина диагонали тетрамино.
/// @param stiff Коэффициент жесткости пружин тетрамино.
/// @param damp Коэффициент демпфирования пружин тетрамино.
void create_L_tetramino(
    std::vector<Particle>& particles, 
    std::vector<Spring>& springs, 
    const sf::Color color,
    float x0, float y0, 
    const float& radius, 
    const float& mass, 
    const float& edge, 
    const float& diagonal, 
    const float& stiff, 
    const float& damp) {

    std::size_t i = particles.size();

    particles.push_back(Particle(x0, y0, radius, mass, color)); // particle_0
    particles.push_back(Particle(x0 + 1.0f * edge, y0, radius, mass, color)); // particle_1
    particles.push_back(Particle(x0 + 2.0f * edge, y0, radius, mass, color)); // particle_2
    particles.push_back(Particle(x0, y0 + 1.0f * edge, radius, mass, color)); // particle_3
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_4
    particles.push_back(Particle(x0 + 2.0f * edge, y0 + 1.0f * edge, radius, mass, color)); // particle_5
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_6
    particles.push_back(Particle(x0 + 2.0f * edge, y0 + 2.0f * edge, radius, mass, color)); // particle_7
    particles.push_back(Particle(x0 + 1.0f * edge, y0 + 3.0f * edge, radius, mass, color)); // particle_8
    particles.push_back(Particle(x0 + 2.0f * edge, y0 + 3.0f * edge, radius, mass, color)); // particle_9

    springs.push_back(Spring(i + 0, i + 1, edge, stiff, damp, color)); // spring_0_1
    springs.push_back(Spring(i + 0, i + 3, edge, stiff, damp, color)); // spring_0_3
    springs.push_back(Spring(i + 0, i + 4, diagonal, stiff, damp, color)); // spring_0_4
    springs.push_back(Spring(i + 1, i + 2, edge, stiff, damp, color)); // spring_1_2
    springs.push_back(Spring(i + 1, i + 3, diagonal, stiff, damp, color)); // spring_1_3
    springs.push_back(Spring(i + 1, i + 4, edge, stiff, damp, color)); // spring_1_4
    springs.push_back(Spring(i + 1, i + 5, diagonal, stiff, damp, color)); // spring_1_5
    springs.push_back(Spring(i + 2, i + 4, diagonal, stiff, damp, color)); // spring_2_4
    springs.push_back(Spring(i + 2, i + 5, edge, stiff, damp, color)); // spring_2_5
    springs.push_back(Spring(i + 3, i + 4, edge, stiff, damp, color)); // spring_3_4
    springs.push_back(Spring(i + 4, i + 5, edge, stiff, damp, color)); // spring_4_5
    springs.push_back(Spring(i + 4, i + 6, edge, stiff, damp, color)); // spring_4_6
    springs.push_back(Spring(i + 4, i + 7, diagonal, stiff, damp, color)); // spring_4_7
    springs.push_back(Spring(i + 5, i + 6, diagonal, stiff, damp, color)); // spring_5_6
    springs.push_back(Spring(i + 5, i + 7, edge, stiff, damp, color)); // spring_5_7
    springs.push_back(Spring(i + 6, i + 7, edge, stiff, damp, color)); // spring_6_7
    springs.push_back(Spring(i + 6, i + 8, edge, stiff, damp, color)); // spring_6_8
    springs.push_back(Spring(i + 6, i + 9, diagonal, stiff, damp, color)); // spring_6_9
    springs.push_back(Spring(i + 7, i + 8, diagonal, stiff, damp, color)); // spring_7_8
    springs.push_back(Spring(i + 7, i + 9, edge, stiff, damp, color)); // spring_7_9
    springs.push_back(Spring(i + 8, i + 9, edge, stiff, damp, color)); // spring_8_9

}


/// @brief Создает стакан и добавляет его полигоны в соответствующий вектор.
/// @param polygons Вектор полигонов для добавления созданных полигонов стакана.
/// @param width Ширина окна.
/// @param height Высота окна.
/// @param glass_width Ширина стенок стакана.
/// @param glass_height Высота стакана.
void create_glass(
    std::vector<Polygon>& polygons, 
    const int& width, 
    const int& height, 
    const float& glass_width, 
    const float& glass_height) {
    
    float thickness = 30.0f;
    float center_x = width / 2;

    // Floor.
    polygons.push_back(Polygon(
        {
            Vector(center_x -  glass_width / 2.0f, height - thickness),
            Vector(center_x -  glass_width / 2.0f, height),
            Vector(center_x +  glass_width / 2.0f, height),
            Vector(center_x +  glass_width / 2.0f, height - thickness)
        },
        Vector(1.0f, 0.0f))
    );

    // Left wall.
    polygons.push_back(Polygon(
        {
            Vector(center_x -  glass_width / 2.0f, height),
            Vector(center_x -  glass_width / 2.0f, height - glass_height),
            Vector(center_x -  glass_width / 2.0f - thickness, height - glass_height),
            Vector(center_x -  glass_width / 2.0f - thickness, height)
        },
        Vector(0.0f, 1.0f))
    );

    // Right wall.
    polygons.push_back(Polygon(
        {
            Vector(center_x +  glass_width / 2.0f, height),
            Vector(center_x +  glass_width / 2.0f, height - glass_height),
            Vector(center_x +  glass_width / 2.0f + thickness, height - glass_height),
            Vector(center_x +  glass_width / 2.0f + thickness, height)
        },
        Vector(0.0f, 1.0f))
    );

}


/// @brief Отрисовка модели.
/// @param polygons Вектор полигонов модели.
/// @param particles Вектор частиц модели.
/// @param springs Вектор пружин модели.
/// @param window Окно для отрисовки.
void draw_model(
    std::vector<Polygon>& polygons, 
    std::vector<Particle>& particles, 
    std::vector<Spring>& springs, 
    sf::RenderWindow& window) {
    
    for (const Spring& spring : springs) spring.draw(window, particles);

    for (const Particle& particle : particles) particle.draw(window);

    for (const Polygon& polygon : polygons) polygon.draw(window);
    
}


/// @brief Выполняет физические вычисления модели.
/// @param polygons Вектор полигонов модели.
/// @param particles Вектор частиц модели.
/// @param springs Вектор пружин модели.
void physics_calculation(
    std::vector<Polygon>& polygons, 
    std::vector<Particle>& particles, 
    std::vector<Spring>& springs) {

    for (Spring& spring : springs) {
        spring.update(particles);
    }

    for (std::size_t i = 0; i < particles.size(); ++i) {
        particles[i].update();

        for (const Polygon& polygon : polygons) {
            bool isCollideWall = particles[i].isCollideWall(polygon);

            if(isCollideWall) particles[i].collideWall(polygon);
        }

        for (std::size_t j = 0; j < particles.size(); ++j) {
            if(i == j) continue;

            bool isCollideParticle = particles[i].isCollideParticle(particles[j]);

            if(isCollideParticle) particles[i].collideParticle(particles[j]);
        }
    }
}


/// @brief Создает взрыв в конце симуляции.
/// @param springs Вектор пружин, которые будут удалены.
void explosion(std::vector<Spring>& springs) {
    springs.clear();
}


int main() {
    // Константы и параметры.
    const int waitTime = 25;
    const int WIDTH = 900;
    const int HEIGHT = 700;
    const float radius = 7.0f;
    const float mass = 3.0f;
    const float edge = 3.3f * radius;
    const float diagonal = edge * std::sqrt(2);
    const float stiff = 7777.7f;
    const float damp = 0.7f;
    const float step = edge + 2.0f * radius;
    const float glass_width = 8.0f * step - 2.0f * radius;
    const float glass_height = 12.0f * step;

    // Цвета для тетрамино.
    const std::vector<sf::Color> colors = {
        sf::Color(255, 0, 0),     // Красный
        sf::Color(128, 0, 0),     // Бордовый
        sf::Color(255, 140, 0),   // Темно-оранжевый
        sf::Color(255, 69, 0),    // Оранжевый
        sf::Color(0, 255, 0),     // Зеленый
        sf::Color(0, 0, 128),     // Синий
        sf::Color(128, 0, 128),   // Фиолетовый
    };

    // Начальные позиции для тетрамино.
    const float y0 = 30.0f;
    const float X0 = WIDTH / 2 - glass_width / 2.0f + radius;
    const std::vector<float> x_positions = {
        X0 + 3 * step - 0.5f,   // x0
        X0 + 2 * step - 1.5f * radius,   // x1
        X0,   // x2
        X0 + 7 * step - 6 * radius,   // x3
        X0 + 6 * step - 6 * radius,   // x4
        X0 + 4 * step - 4 * radius,   // x5
        X0 + step - 2.0f * radius,   // x6
        X0 + 4 * step - 4 * radius,   // x7
        X0 + 6 * step - 6 * radius,   // x8
        X0,   // x9
        X0 + 3 * step - 3 * radius,   // x10
        X0 + 2.0f * radius   // x11
    };

    // Флаги для создания новых тетрамино.
    std::vector<bool> permissions(11, true);

    // Основные массивы.
    std::vector<Polygon> polygons;
    std::vector<Particle> particles;
    std::vector<Spring> springs;

    create_glass(polygons, WIDTH, HEIGHT, glass_width, glass_height);
    create_O_tetramino(particles, springs, colors[0], x_positions[0], y0, radius, mass, edge, diagonal, stiff, damp);
    
    // Создаем окно размером WIDTH x HEIGHT пикселей.
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Soft tetris");

    // Создаем часы для измерения времени.
    sf::Clock clock;
    sf::Clock timer;
    timer.restart();

    // Главный цикл приложения.
    while (window.isOpen()) {
        // Обработка событий.
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        // Очистка экрана.
        window.clear(sf::Color(47, 79, 79)); // Темно-серый

        // Создание новых тетрамино по таймеру.
        for (std::size_t i = 0; i < permissions.size(); ++i) {
            if (permissions[i] && timer.getElapsedTime() > sf::seconds((0.5f * i + 1) * waitTime)) {
                permissions[i] = false;
                switch (i) {
                    case 1:
                        create_T_tetramino(particles, springs, colors[i % 7], x_positions[i], y0, radius, mass, edge, diagonal, stiff, damp); // 1
                        break;
                    case 2:
                        create_T_tetramino(particles, springs, colors[i % 7], x_positions[i], y0, radius, mass, edge, diagonal, stiff, damp); // 2
                        break;
                    case 3:
                        create_O_tetramino(particles, springs, colors[i % 7], x_positions[i], y0, radius, mass, edge, diagonal, stiff, damp); // 3
                        break;
                    case 4:
                        create_I1_tetramino(particles, springs, colors[i % 7], x_positions[i], y0, radius, mass, edge, diagonal, stiff, damp); // 4
                        break;
                    case 5:
                        create_Z1_tetramino(particles, springs, colors[i % 7], x_positions[i], y0, radius, mass, edge, diagonal, stiff, damp); // 5
                        break;
                    case 6:
                        create_J2_tetramino(particles, springs, colors[i % 7], x_positions[i], y0, radius, mass, edge, diagonal, stiff, damp); // 6
                        break;
                    case 7:
                        create_O_tetramino(particles, springs, colors[i % 7], x_positions[i], y0, radius, mass, edge, diagonal, stiff, damp); // 7
                        break;
                    case 8:
                        create_Z2_tetramino(particles, springs, colors[i % 7], x_positions[i], y0, radius, mass, edge, diagonal, stiff, damp); // 8
                        break;
                    case 9:
                        create_T_tetramino(particles, springs, colors[i % 7], x_positions[i], y0, radius, mass, edge, diagonal, stiff, damp); // 9
                        break;
                    case 10:
                        create_I2_tetramino(particles, springs, colors[i % 7], x_positions[i], y0, radius, mass, edge, diagonal, stiff, damp); // 10
                        break;
                    case 11:
                        create_L_tetramino(particles, springs, colors[i % 7], x_positions[i], y0, radius, mass, edge, diagonal, stiff, damp); // 11
                        break;
                    default:
                        break;
                }
            }
        }

        // Взрыв.
        if(timer.getElapsedTime() > sf::seconds(6.7f * waitTime)) {
            explosion(springs);
        }

        // Обработка физики модели.
        physics_calculation(polygons, particles, springs);

        // Задержка для установление определенной частоты обновленя кадров. 
        sf::sleep(sf::seconds(dt));

        // Отрисовка модели.
        draw_model(polygons, particles, springs, window);

        // Отображение результата.
        window.display();

        // Сброс часов.
        clock.restart();

    }

    return 0;  
}
