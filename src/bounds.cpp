#include "bounds.hpp"
#include "population.hpp"
#include "parameters.hpp"

static double modulo2(const int x)
{
	return (2 + x % 2) % 2;
}

namespace bounds
{
	Mask BoundCorrection::is_out_of_bounds(const Vector& xi) const
	{
		return xi.array() < lb.array() || xi.array() > ub.array();
	}


	Vector BoundCorrection::delta_out_of_bounds(const Vector& xi, const Mask& oob) const
	{
		return 	(oob).select((xi - lb).cwiseQuotient(db), xi);;
	}



	void BoundCorrection::correct(const Eigen::Index i, parameters::Parameters& p)
	{
		const auto oob = is_out_of_bounds(p.pop.X.col(i));
		if (oob.any())
		{
			n_out_of_bounds++;
			p.pop.X.col(i) = correct_x(p.pop.X.col(i), oob);
			p.pop.Y.col(i) = p.adaptation->invert_x(p.pop.X.col(i), p.mutation->sigma);
			p.pop.Z.col(i) = p.adaptation->invert_y(p.pop.Y.col(i));
		}
	}


	Vector COTN::correct_x(const Vector& xi, const Mask& oob)
	{
		const Vector y = delta_out_of_bounds(xi, oob);
		return (oob).select(
			lb.array() + db.array() * ((y.array() > 0).cast<double>() - sampler().array().abs()).abs(), y);
	}


	Vector Mirror::correct_x(const Vector& xi, const Mask& oob)
	{
		const Vector y = delta_out_of_bounds(xi, oob);
		return (oob).select(
			lb.array() + db.array() * (y.array() - y.array().floor() - y.array().floor().unaryExpr(&modulo2)).
			abs(),
			y);
	}


	Vector UniformResample::correct_x(const Vector& xi, const Mask& oob)
	{
		return (oob).select(lb + sampler().cwiseProduct(db), xi);
	}

	Vector Saturate::correct_x(const Vector& xi, const Mask& oob)
	{
		const Vector y = delta_out_of_bounds(xi, oob);
		return (oob).select(
			lb.array() + db.array() * (y.array() > 0).cast<double>(), y);
	}

	Vector Toroidal::correct_x(const Vector& xi, const Mask& oob)
	{
		const Vector y = delta_out_of_bounds(xi, oob);
		return (oob).select(
			lb.array() + db.array() * (y.array() - y.array().floor()).abs(), y);
	}
}
