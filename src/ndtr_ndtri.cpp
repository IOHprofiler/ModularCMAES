#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include "common.hpp"

/*                                                  adopted from:  ndtri.c
 *
 *     Inverse of Normal distribution function
 *
 *
 *
 * SYNOPSIS:
 *
 * Float x, y, ndtri();
 *
 * x = ndtri( y );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the argument, x, for which the area under the
 * Gaussian probability density function (integrated from
 * minus infinity to x) is equal to y.
 *
 *
 * For small arguments 0 < y < exp(-2), the program computes
 * z = sqrt( -2.0 * log(y) );  then the approximation is
 * x = z - log(z)/z  - (1/z) P(1/z) / Q(1/z).
 * There are two rational functions P/Q, one for 0 < y < exp(-32)
 * and the other for y up to exp(-2).  For larger arguments,
 * w = y - 0.5, and  x/sqrt(2pi) = w + w**3 R(w**2)/S(w**2)).
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain        # trials      peak         rms
 *    IEEE     0.125, 1        20000       7.2e-16     1.3e-16
 *    IEEE     3e-308, 0.135   50000       4.6e-16     9.8e-17
 *
 *
 * ERROR MESSAGES:
 *
 *   message         condition    value returned
 * ndtri domain       x < 0        NPY_NAN
 * ndtri domain       x > 1        NPY_NAN
 *
 */

/*
 * Cephes Math Library Release 2.1:  January, 1989
 * Copyright 1984, 1987, 1989 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 * 
*/

/* sqrt(2pi) */
static Float s2pi = 2.50662827463100050242E0;

/* approximation for 0 <= |y - 0.5| <= 3/8 */
static Float P0[5] = {
	-5.99633501014107895267E1,
	9.80010754185999661536E1,
	-5.66762857469070293439E1,
	1.39312609387279679503E1,
	-1.23916583867381258016E0,
};

static Float Q0[8] = {
	/* 1.00000000000000000000E0, */
	1.95448858338141759834E0,
	4.67627912898881538453E0,
	8.63602421390890590575E1,
	-2.25462687854119370527E2,
	2.00260212380060660359E2,
	-8.20372256168333339912E1,
	1.59056225126211695515E1,
	-1.18331621121330003142E0,
};

/* Approximation for interval z = sqrt(-2 log y ) between 2 and 8
 * i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
 */
static Float P1[9] = {
	4.05544892305962419923E0,
	3.15251094599893866154E1,
	5.71628192246421288162E1,
	4.40805073893200834700E1,
	1.46849561928858024014E1,
	2.18663306850790267539E0,
	-1.40256079171354495875E-1,
	-3.50424626827848203418E-2,
	-8.57456785154685413611E-4,
};

static Float Q1[8] = {
	/*  1.00000000000000000000E0, */
	1.57799883256466749731E1,
	4.53907635128879210584E1,
	4.13172038254672030440E1,
	1.50425385692907503408E1,
	2.50464946208309415979E0,
	-1.42182922854787788574E-1,
	-3.80806407691578277194E-2,
	-9.33259480895457427372E-4,
};

/* Approximation for interval z = sqrt(-2 log y ) between 8 and 64
 * i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
 */

static Float P2[9] = {
	3.23774891776946035970E0,
	6.91522889068984211695E0,
	3.93881025292474443415E0,
	1.33303460815807542389E0,
	2.01485389549179081538E-1,
	1.23716634817820021358E-2,
	3.01581553508235416007E-4,
	2.65806974686737550832E-6,
	6.23974539184983293730E-9,
};

static Float Q2[8] = {
	/*  1.00000000000000000000E0, */
	6.02427039364742014255E0,
	3.67983563856160859403E0,
	1.37702099489081330271E0,
	2.16236993594496635890E-1,
	1.34204006088543189037E-2,
	3.28014464682127739104E-4,
	2.89247864745380683936E-6,
	6.79019408009981274425E-9,
};

static Float polevl(Float x, const Float coef[], int N)
{
	Float ans;
	int i;
	const Float* p;

	p = coef;
	ans = *p++;
	i = N;

	do
	{
		ans = ans * x + *p++;
	}
	while (--i);

	return (ans);
}

static Float p1evl(Float x, const Float coef[], int N)
{
	Float ans;
	const Float* p;
	int i;

	p = coef;
	ans = x + *p++;
	i = N - 1;

	do
	{
		ans = ans * x + *p++;
	}
	while (--i);

	return (ans);
}

static Float ndtri(Float y0)
{
	Float x, y, z, y2, x0, x1;
	int code;

	if (y0 == 0.0)
	{
		return -std::numeric_limits<Float>::infinity();
	}
	if (y0 == 1.0)
	{
		return std::numeric_limits<Float>::infinity();
	}
	if (y0 < 0.0 || y0 > 1.0)
	{
		return std::numeric_limits<Float>::signaling_NaN();
	}
	code = 1;
	y = y0;
	if (y > (1.0 - 0.13533528323661269189))
	{
		/* 0.135... = exp(-2) */
		y = 1.0 - y;
		code = 0;
	}

	if (y > 0.13533528323661269189)
	{
		y = y - 0.5;
		y2 = y * y;
		x = y + y * (y2 * polevl(y2, P0, 4) / p1evl(y2, Q0, 8));
		x = x * s2pi;
		return (x);
	}

	x = sqrt(-2.0 * std::log(y));
	x0 = x - log(x) / x;

	z = 1.0 / x;
	if (x < 8.0) /* y > exp(-32) = 1.2664165549e-14 */
		x1 = z * polevl(z, P1, 8) / p1evl(z, Q1, 8);
	else
		x1 = z * polevl(z, P2, 8) / p1evl(z, Q2, 8);
	x = x0 - x1;
	if (code != 0)
		x = -x;
	return (x);
}


Float ppf(const Float x) { return ndtri(x); }

/*
                                                    Adopted from ndtr.c
 *
 *     Normal distribution function
 *
 *
 *
 * SYNOPSIS:
 *
 * Float x, y, ndtr();
 *
 * y = ndtr( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the area under the Gaussian probability density
 * function, integrated from minus infinity to x:
 *
 *                            x
 *                             -
 *                   1        | |          2
 *    ndtr(x)  = ---------    |    exp( - t /2 ) dt
 *               sqrt(2pi)  | |
 *                           -
 *                          -inf.
 *
 *             =  ( 1 + erf(z) ) / 2
 *             =  erfc(z) / 2
 *
 * where z = x/sqrt(2). Computation is via the functions
 * erf and erfc.
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE     -13,0        30000       3.4e-14     6.7e-15
 *
 *
 * ERROR MESSAGES:
 *
 *   message         condition         value returned
 * erfc underflow    x > 37.519379347       0.0
 *
 */
/*							erf.c
 *
 *	Error function
 *
 *
 *
 * SYNOPSIS:
 *
 * Float x, y, erf();
 *
 * y = erf( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * The integral is
 *
 *                           x
 *                            -
 *                 2         | |          2
 *   erf(x)  =  --------     |    exp( - t  ) dt.
 *              sqrt(pi)   | |
 *                          -
 *                           0
 *
 * For 0 <= |x| < 1, erf(x) = x * P4(x**2)/Q5(x**2); otherwise
 * erf(x) = 1 - erfc(x).
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,1         30000       3.7e-16     1.0e-16
 *
 */
/*							erfc.c
 *
 *	Complementary error function
 *
 *
 *
 * SYNOPSIS:
 *
 * Float x, y, erfc();
 *
 * y = erfc( x );
 *
 *
 *
 * DESCRIPTION:
 *
 *
 *  1 - erf(x) =
 *
 *                           inf.
 *                             -
 *                  2         | |          2
 *   erfc(x)  =  --------     |    exp( - t  ) dt
 *               sqrt(pi)   | |
 *                           -
 *                            x
 *
 *
 * For small x, erfc(x) = 1 - erf(x); otherwise rational
 * approximations are computed.
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,26.6417   30000       5.7e-14     1.5e-14
 */


/*
 * Cephes Math Library Release 2.2:  June, 1992
 * Copyright 1984, 1987, 1988, 1992 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

static Float P[] = {
    2.46196981473530512524E-10,
    5.64189564831068821977E-1,
    7.46321056442269912687E0,
    4.86371970985681366614E1,
    1.96520832956077098242E2,
    5.26445194995477358631E2,
    9.34528527171957607540E2,
    1.02755188689515710272E3,
    5.57535335369399327526E2
};

static Float Q[] = {
    /* 1.00000000000000000000E0, */
    1.32281951154744992508E1,
    8.67072140885989742329E1,
    3.54937778887819891062E2,
    9.75708501743205489753E2,
    1.82390916687909736289E3,
    2.24633760818710981792E3,
    1.65666309194161350182E3,
    5.57535340817727675546E2
};

static Float R[] = {
    5.64189583547755073984E-1,
    1.27536670759978104416E0,
    5.01905042251180477414E0,
    6.16021097993053585195E0,
    7.40974269950448939160E0,
    2.97886665372100240670E0
};

static Float S[] = {
    /* 1.00000000000000000000E0, */
    2.26052863220117276590E0,
    9.39603524938001434673E0,
    1.20489539808096656605E1,
    1.70814450747565897222E1,
    9.60896809063285878198E0,
    3.36907645100081516050E0
};

static Float T[] = {
    9.60497373987051638749E0,
    9.00260197203842689217E1,
    2.23200534594684319226E3,
    7.00332514112805075473E3,
    5.55923013010394962768E4
};

static Float U[] = {
    /* 1.00000000000000000000E0, */
    3.35617141647503099647E1,
    5.21357949780152679795E2,
    4.59432382970980127987E3,
    2.26290000613890934246E4,
    4.92673942608635921086E4
};

#define UTHRESH 37.519379347
#define MAXLOG 7.09782712893383996843E2

static Float ndtri_erf(Float x);
static Float ndtri_erfc(Float a)
{
    Float p, q, x, y, z;

    if (std::isnan(a))
		return std::numeric_limits<Float>::signaling_NaN();

    if (a < 0.0) {
        x = -a;
    }
    else {
        x = a;
    }

    if (x < 1.0) {
        return 1.0 - ndtri_erf(a);
    }

    z = -a * a;

    if (z < -MAXLOG) {
        goto under;
    }

    z = exp(z);

    if (x < 8.0) {
        p = polevl(x, P, 8);
        q = p1evl(x, Q, 8);
    }
    else {
        p = polevl(x, R, 5);
        q = p1evl(x, S, 6);
    }
    y = (z * p) / q;

    if (a < 0) {
        y = 2.0 - y;
    }

    if (y != 0.0) {
        return y;
    }

under:
    if (a < 0) {
        return 2.0;
    }
    else {
        return 0.0;
    }
}

static Float ndtri_erf(Float x)
{
    Float y, z;

    if (std::isnan(x))
		return std::numeric_limits<Float>::signaling_NaN();

    if (x < 0.0) {
        return -ndtri_erf(-x);
    }

    if (fabs(x) > 1.0) {
        return (1.0 - ndtri_erfc(x));
    }
    z = x * x;

    y = x * polevl(z, T, 4) / p1evl(z, U, 5);
    return y;
}

static Float ndtr(Float a)
{
    Float x, y, z;

	if (std::isnan(a))
		return std::numeric_limits<Float>::signaling_NaN();

    x = a * M_SQRT1_2;
    z = fabs(x);

    if (z < M_SQRT1_2) {
        y = 0.5 + 0.5 * ndtri_erf(x);
    }
    else {
        y = 0.5 * ndtri_erfc(z);
        if (x > 0) {
            y = 1.0 - y;
        }
    }

    return y;
}

Float cdf(const Float x) { return ndtr(x); }