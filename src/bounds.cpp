#include "bounds.hpp"
#include "population.hpp"

namespace bounds
{
    void CountOutOfBounds::correct(Population& pop, const Vector& m)
    {
        n_out_of_bounds = 0;
        for (auto i = 0; i < pop.X.cols(); ++i)
        {
            const auto oob = pop.X.col(i).array() < lb.array() || pop.X.col(i).array() > ub.array();
            n_out_of_bounds += oob.sum(); 
        }
    }

    double modulo2(const int x)
    {
        return static_cast<double>((2 + (x % 2)) % 2);
    };

    void COTN::correct(Population& pop, const Vector& m)
    {
        n_out_of_bounds = 0;

        for (auto i = 0; i < pop.X.cols(); ++i)
        {
            const auto oob = pop.X.col(i).array() < lb.array() || pop.X.col(i).array() > ub.array();
            if (oob.any())
            {
                n_out_of_bounds++;
                const Vector y = (oob).select((pop.X.col(i) - lb).cwiseQuotient(db), pop.X.col(i));
                pop.X.col(i) = (oob).select(
                    lb.array() + db.array() * ((y.array() > 0).cast<double>() - sampler().array().abs()).abs(), y);
                pop.Y.col(i) = (pop.X.col(i) - m) / pop.s(i);
            }
        }
        
    }

    void Mirror::correct(Population& pop, const Vector& m)
    {
        n_out_of_bounds = 0;

        for (auto i = 0; i < pop.X.cols(); ++i)
        {
            const auto oob = pop.X.col(i).array() < lb.array() || pop.X.col(i).array() > ub.array();
            if (oob.any())
            {
                n_out_of_bounds++;
                const Vector y = (oob).select((pop.X.col(i) - lb).cwiseQuotient(db), pop.X.col(i));
                pop.X.col(i) = (oob).select(
                    lb.array() + db.array() * (y.array() - y.array().floor() - y.array().floor().unaryExpr(&modulo2)).abs(),
                    y);
                pop.Y.col(i) = (pop.X.col(i) - m) / pop.s(i);
            }
        }
        
    }

    void UniformResample::correct(Population& pop, const Vector& m)
    {
        n_out_of_bounds = 0;

        for (auto i = 0; i < pop.X.cols(); ++i)
        {
            const auto oob = pop.X.col(i).array() < lb.array() || pop.X.col(i).array() > ub.array();
            if (oob.any())
            {
                n_out_of_bounds++;
                pop.X.col(i) = (oob).select(lb + sampler().cwiseProduct(db), pop.X.col(i));
                pop.Y.col(i) = (pop.X.col(i) - m) / pop.s(i);
            }
        }
        
    } 

    void Saturate::correct(Population& pop, const Vector& m)
    {
        n_out_of_bounds = 0;

        for (auto i = 0; i < pop.X.cols(); ++i)
        {
            const auto oob = pop.X.col(i).array() < lb.array() || pop.X.col(i).array() > ub.array();
            if (oob.any())
            {
                n_out_of_bounds++;
                const Vector y = (oob).select((pop.X.col(i) - lb).cwiseQuotient(db), pop.X.col(i));
                pop.X.col(i) = (oob).select(
                    lb.array() + db.array() * (y.array() > 0).cast<double>(), y);
                pop.Y.col(i) = (pop.X.col(i) - m) / pop.s(i);
            }
        }
        
    }

    void Toroidal::correct(Population& pop, const Vector& m)
    {
        n_out_of_bounds = 0;

        for (auto i = 0; i < pop.X.cols(); ++i)
        {
            const auto oob = pop.X.col(i).array() < lb.array() || pop.X.col(i).array() > ub.array();
            if (oob.any())
            {
                n_out_of_bounds++;
                const Vector y = (oob).select((pop.X.col(i) - lb).cwiseQuotient(db), pop.X.col(i));
                pop.X.col(i) = (oob).select(
                    lb.array() + db.array() * (y.array() - y.array().floor()).abs(), y);
                pop.Y.col(i) = (pop.X.col(i) - m) / pop.s(i);
            }
        }
        
    }

}