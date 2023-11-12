#include "c_maes.hpp"
#include <chrono>


struct Function {
    size_t evals = 0;
    double operator()(const Vector& x)
    {
        double res = x.dot(x);
        evals++;
        return res;
    }
};




template <typename Callable>
void call(Callable& o)
{
    static_assert(std::is_invocable_r<double, Callable, Vector>::value, "Incorrect objective function type");
    double result = o(Vector::Ones(10));
    std::cout << result;
}
 



int main() {
    using namespace std::placeholders;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;



    size_t dim = 40;
    parameters::Settings s(dim);
    s.budget = 10'000 * dim;
    s.modules.ssa = parameters::StepSizeAdaptation::PSR;
    auto p = std::make_shared<parameters::Parameters>(s);
    
    Function f;

    ModularCMAES cma(p);

    FunctionType func = std::bind(&Function::operator(), &f, _1);
    auto t1 = high_resolution_clock::now();
    cma(func);
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_int.count() << "ms\n";
    std::cout << "completed\n";
    std::cout << p->stats.evaluations << ", " << f.evals << std::endl;
    std::cin.get();

}