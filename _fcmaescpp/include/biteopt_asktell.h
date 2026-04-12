//$ nocpp
//$ noclang

/**
 * @file biteopt_asktell.h
 *
 * @brief Ask/tell interface for BiteOpt optimizer.
 *
 * Allows batch generation of candidate solutions for external evaluation.
 * Instead of calling optcost() inside the optimizer, users fetch candidates
 * via ask(), evaluate them elsewhere, and report costs back via tell().
 *
 * Within a batch, candidates are generated from a frozen optimizer state.
 * Population updates and selector feedback happen during tell().
 */

#ifndef BITEOPT_ASKTELL_INCLUDED
#define BITEOPT_ASKTELL_INCLUDED

#include "biteopt.h"
#include <algorithm>
#include <cstring>

/**
 * Single-level BiteOpt optimizer with ask/tell capability.
 * Subclasses CBiteOpt to access protected generation machinery.
 */

class CBiteOptAT : public CBiteOpt
{
	friend class CBiteOptDeepAT;

public:
	CBiteOptAT()
		: AskParamsBuf( NULL )
		, AskValuesBuf( NULL )
		, AskSelBuf( NULL )
		, AskSelCounts( NULL )
		, MaxBatch( 0 )
	{
	}

	virtual ~CBiteOptAT()
	{
		deleteAskBuffers();
	}

	/**
	 * Keeps ask buffers consistent with the currently configured dimensions.
	 */

	void updateDims( const int aParamCount, const int PopSize0 = 0 )
	{
		const int PrevParamCount = ParamCount;
		const int PrevPopSize = PopSize;

		CBiteOpt::updateDims( aParamCount, PopSize0 );

		if( PrevParamCount != ParamCount || PrevPopSize != PopSize )
		{
			deleteAskBuffers();
		}
	}

	/**
	 * Allocate internal buffers for batch ask/tell.
	 *
	 * @param aMaxBatch Maximum batch size that will be requested.
	 */

	void initAskBuffers( const int aMaxBatch )
	{
		if( aMaxBatch <= 0 || aMaxBatch <= MaxBatch )
		{
			return;
		}

		deleteAskBuffers();

		MaxBatch = aMaxBatch;
		AskParamsBuf = new ptype[ MaxBatch * ParamCount ];
		AskValuesBuf = new double[ MaxBatch * ParamCount ];
		AskSelBuf = new CBiteSelBase*[ MaxBatch * MaxApplySels ];
		AskSelCounts = new int[ MaxBatch ];
	}

	/**
	 * Generate a single candidate solution without evaluation.
	 * Writes to slot `idx` in the ask buffers.
	 *
	 * @param rnd PRNG object.
	 * @param idx Slot index in the ask buffers (0-based).
	 * @return Pointer to real-valued parameter vector.
	 */

	const double* askOne( CBiteRnd& rnd, const int idx )
	{
		ptype* const sp = AskParamsBuf + idx * ParamCount;
		double* const sv = AskValuesBuf + idx * ParamCount;
		int i;

		if( DoInitEvals )
		{
			const ptype* const Params = getCurParams();

			for( i = 0; i < ParamCount; i++ )
			{
				sv[ i ] = getRealValue( Params, i );
			}

			memcpy( sp, Params, ParamCount * sizeof( ptype ));
			AskSelCounts[ idx ] = 0;

			return( sv );
		}

		ApplySelsCount = 0;

		int SelMethod = select( MethodSel, rnd );

		// Method 3 uses embedded optimizers that evaluate optcost() internally.
		// Re-sample MethodSel until one of the ask/tell-compatible generators
		// is chosen, while keeping only a single MethodSel entry in ApplySels.

		while( SelMethod == 3 )
		{
			SelMethod = MethodSel.select( rnd );
		}

		if( SelMethod == 0 )
		{
			generateSol2( rnd );
		}
		else
		if( SelMethod == 1 )
		{
			const int SelM1 = select( M1Sel, rnd );

			if( SelM1 == 0 )
			{
				const int SelM1A = select( M1ASel, rnd );

				if( SelM1A == 0 )
				{
					generateSol2b( rnd );
				}
				else
				if( SelM1A == 1 )
				{
					generateSol2c( rnd );
				}
				else
				{
					generateSol2d( rnd );
				}
			}
			else
			if( SelM1 == 1 )
			{
				if( select( M1BSel, rnd ))
				{
					generateSol4( rnd );
				}
				else
				{
					generateSol5b( rnd );
				}
			}
			else
			if( SelM1 == 2 )
			{
				if( select( M1CSel, rnd ))
				{
					generateSol5( rnd );
				}
				else
				{
					generateSol10( rnd );
				}
			}
			else
			{
				generateSol6( rnd );
			}
		}
		else
		{
			if( select( M2Sel, rnd ))
			{
				generateSol1( rnd );
			}
			else
			{
				const int SelM2B = select( M2BSel, rnd );

				if( SelM2B == 0 )
				{
					generateSol3( rnd );
				}
				else
				if( SelM2B == 1 )
				{
					generateSol7( rnd );
				}
				else
				if( SelM2B == 2 )
				{
					generateSol8( rnd );
				}
				else
				{
					generateSol9( rnd );
				}
			}
		}

		for( i = 0; i < ParamCount; i++ )
		{
			TmpParams[ i ] = wrapParam( rnd, TmpParams[ i ]);
			NewValues[ i ] = getRealValue( TmpParams, i );
		}

		memcpy( sp, TmpParams, ParamCount * sizeof( ptype ));
		memcpy( sv, NewValues, ParamCount * sizeof( double ));
		memcpy( AskSelBuf + idx * MaxApplySels, ApplySels,
			ApplySelsCount * sizeof( CBiteSelBase* ));

		AskSelCounts[ idx ] = ApplySelsCount;

		return( sv );
	}

	/**
	 * Feed back the cost for candidate `idx`, updating population and
	 * selector state.
	 *
	 * @param rnd PRNG object.
	 * @param idx Slot index matching the prior askOne() call.
	 * @param cost Objective value for this candidate.
	 * @param PushOpt Optional push-optimizer for Deep mode.
	 * @return StallCount (0 if this solution was accepted).
	 */

	int tellOne( CBiteRnd& rnd, const int idx, const double cost,
		CBiteOptAT* const PushOpt = NULL )
	{
		const ptype* const sp = AskParamsBuf + idx * ParamCount;
		const double* const sv = AskValuesBuf + idx * ParamCount;
		const double fc = sanitizeCost( cost );
		int i;

		if( DoInitEvals )
		{
			ptype* const Params = getCurParams();
			memcpy( Params, sp, ParamCount * sizeof( ptype ));
			memcpy( NewValues, sv, ParamCount * sizeof( double ));

			NewCost = fc;
			updateBestCost( NewCost, NewValues,
				updatePop( NewCost, Params, false ));

			if( CurPopPos == PopSize )
			{
				updateCentroid();

				for( i = 0; i < ParPopCount; i++ )
				{
					ParPops[ i ] -> copy( *this );
				}

				DoInitEvals = false;
			}

			return( 0 );
		}

		ApplySelsCount = AskSelCounts[ idx ];
		memcpy( ApplySels, AskSelBuf + idx * MaxApplySels,
			ApplySelsCount * sizeof( CBiteSelBase* ));

		memcpy( TmpParams, sp, ParamCount * sizeof( ptype ));
		memcpy( NewValues, sv, ParamCount * sizeof( double ));

		NewCost = fc;

		const int p = updatePop( NewCost, TmpParams, true );

		if( p > CurPopSize1 )
		{
			applySelsDecr( rnd );
			StallCount++;

			if( CurPopSize < PopSize )
			{
				if( select( PopChangeIncrSel, rnd ))
				{
					incrCurPopSize();
				}
			}
		}
		else
		{
			updateBestCost( NewCost, NewValues, p );
			applySelsIncr( rnd, 1.0 - p * CurPopSizeI );

			StallCount = 0;

			if( rnd.get() < ParamCountI )
			{
				OldPop.updatePop( *getObjPtr( PopParams[ CurPopSize1 ]),
					PopParams[ CurPopSize1 ], false );
			}

			if( PushOpt != NULL && PushOpt != this &&
				!PushOpt -> DoInitEvals && p > 0 )
			{
				PushOpt -> updatePop( NewCost, TmpParams, true );
				PushOpt -> updateParPop( NewCost, TmpParams );
			}

			if( CurPopSize > PopSize / 2 )
			{
				if( select( PopChangeDecrSel, rnd ))
				{
					decrCurPopSize();
				}
			}
		}

		updateParPop( NewCost, TmpParams );

		return( StallCount );
	}

	/**
	 * @return Pointer to real-valued parameter vector for candidate `idx`.
	 */

	const double* getAskValues( const int idx ) const
	{
		return( AskValuesBuf + idx * ParamCount );
	}

	bool isInitPhase() const { return( DoInitEvals ); }
	int getPopSize() const { return( PopSize ); }
	int getInitPos() const { return( CurPopPos ); }

protected:
	virtual double optcost( const double* const p )
	{
		(void) p;
		return( 1e300 );
	}

	virtual void getMinValues( double* const p ) const
	{
		(void) p;
	}

	virtual void getMaxValues( double* const p ) const
	{
		(void) p;
	}

private:
	static double sanitizeCost( const double cost )
	{
		return( cost == cost ? cost : 1e300 );
	}

	ptype* AskParamsBuf;
	double* AskValuesBuf;
	CBiteSelBase** AskSelBuf;
	int* AskSelCounts;
	int MaxBatch;

	void deleteAskBuffers()
	{
		delete[] AskParamsBuf;
		delete[] AskValuesBuf;
		delete[] AskSelBuf;
		delete[] AskSelCounts;

		AskParamsBuf = NULL;
		AskValuesBuf = NULL;
		AskSelBuf = NULL;
		AskSelCounts = NULL;
		MaxBatch = 0;
	}
};

/**
 * Ask/tell wrapper at the Deep optimization level.
 * Manages M CBiteOptAT sub-optimizers with Deep-mode solution exchange.
 */

class CBiteOptDeepAT : public CBiteOptInterface
{
public:
	CBiteOptDeepAT()
		: ATParamCount( 0 )
		, OptCount( 0 )
		, Opts( NULL )
		, BestOpt( NULL )
		, CurOpt( NULL )
		, PushOpt( NULL )
		, LastOpt( NULL )
		, ATStallCount( 0 )
		, LowerBounds( NULL )
		, UpperBounds( NULL )
		, CurBatchSize( 0 )
		, CandOptIdx( NULL )
		, SortOrder( NULL )
		, SortCosts( NULL )
		, MaxBatch( 0 )
	{
	}

	virtual ~CBiteOptDeepAT()
	{
		deleteBuffers();
	}

	virtual const double* getBestParams() const
	{
		return( BestOpt -> getBestParams() );
	}

	virtual double getBestCost() const
	{
		return( BestOpt -> getBestCost() );
	}

	CBiteSelBase** getSels()
	{
		return( CurOpt == NULL ? NULL : CurOpt -> getSels() );
	}

	const char** getSelNames() const
	{
		return( CurOpt == NULL ? NULL : CurOpt -> getSelNames() );
	}

	int getSelCount() const
	{
		return( CurOpt == NULL ? 0 : CurOpt -> getSelCount() );
	}

	/**
	 * @param aParamCount Number of parameters being optimized.
	 * @param M Depth (number of sub-optimizers). 1 = plain, >1 = deep.
	 * @param PopSize0 Population size override. 0 = default formula.
	 */

	void updateDims( const int aParamCount, const int M = 1,
		const int PopSize0 = 0 )
	{
		const int NewOptCount = ( M > 0 ? M : 1 );

		if( aParamCount == ATParamCount && NewOptCount == OptCount )
		{
			return;
		}

		deleteBuffers();

		ATParamCount = aParamCount;
		OptCount = NewOptCount;

		LowerBounds = new double[ ATParamCount ];
		UpperBounds = new double[ ATParamCount ];
		Opts = new CBiteOptWrapAT*[ OptCount ];

		for( int i = 0; i < OptCount; i++ )
		{
			Opts[ i ] = new CBiteOptWrapAT( this );
			Opts[ i ] -> updateDims( aParamCount, PopSize0 );
		}
	}

	/**
	 * Initialize with bounds and optional starting point.
	 *
	 * @param rnd PRNG object.
	 * @param lb Lower bounds array.
	 * @param ub Upper bounds array.
	 * @param InitParams Optional initial parameter vector.
	 * @param InitRadius Initial radius multiplier.
	 */

	void init( CBiteRnd& rnd, const double* const lb, const double* const ub,
		const double* const InitParams = NULL,
		const double InitRadius = 1.0 )
	{
		memcpy( LowerBounds, lb, ATParamCount * sizeof( double ));
		memcpy( UpperBounds, ub, ATParamCount * sizeof( double ));

		for( int i = 0; i < OptCount; i++ )
		{
			Opts[ i ] -> init( rnd, InitParams, InitRadius );
		}

		BestOpt = Opts[ 0 ];
		CurOpt = Opts[ 0 ];
		LastOpt = CurOpt;
		ATStallCount = 0;

		if( OptCount == 1 )
		{
			PushOpt = CurOpt;
		}
		else
		{
			while( true )
			{
				PushOpt = Opts[ rnd.getInt( OptCount )];

				if( PushOpt != CurOpt )
				{
					break;
				}
			}
		}
	}

	/**
	 * Generate a batch of candidate solutions for parallel evaluation.
	 *
	 * @param rnd PRNG object.
	 * @param batchSize Desired number of candidates.
	 * @return Actual number of candidates generated.
	 */

	int ask( CBiteRnd& rnd, int batchSize )
	{
		CurBatchSize = 0;

		if( batchSize <= 0 )
		{
			return( 0 );
		}

		ensureBatchBuffers( batchSize );

		if( CurOpt -> isInitPhase() )
		{
			const int remaining = CurOpt -> getPopSize() -
				CurOpt -> getInitPos();

			if( batchSize > remaining )
			{
				batchSize = remaining;
			}
		}

		const int curOptIdx = getOptIndex( CurOpt );

		for( int i = 0; i < batchSize; i++ )
		{
			CurOpt -> askOne( rnd, i );
			CandOptIdx[ i ] = curOptIdx;
			CurBatchSize++;
		}

		return( CurBatchSize );
	}

	/**
	 * Feed back costs for all candidates from the most recent ask().
	 *
	 * Results are processed best-first using sanitized costs so NaNs sort to
	 * the end and do not break std::sort's ordering requirements.
	 *
	 * @param rnd PRNG object.
	 * @param costs Objective values corresponding to getAskValues(i).
	 */

	void tell( CBiteRnd& rnd, const double* costs )
	{
		if( CurBatchSize <= 0 || costs == NULL )
		{
			CurBatchSize = 0;
			return;
		}

		for( int i = 0; i < CurBatchSize; i++ )
		{
			SortOrder[ i ] = i;
			SortCosts[ i ] = sanitizeCost( costs[ i ]);
		}

		const double* const sc = SortCosts;
		std::sort( SortOrder, SortOrder + CurBatchSize,
			[ sc ]( const int a, const int b )
			{
				if( sc[ a ] < sc[ b ] )
				{
					return( true );
				}

				if( sc[ b ] < sc[ a ] )
				{
					return( false );
				}

				return( a < b );
			});

		for( int si = 0; si < CurBatchSize; si++ )
		{
			const int i = SortOrder[ si ];
			CBiteOptWrapAT* const opt = Opts[ CandOptIdx[ i ]];
			CBiteOptAT* const PO = ( OptCount == 1 ?
				(CBiteOptAT*) opt : (CBiteOptAT*) PushOpt );

			const int sc2 = opt -> tellOne( rnd, i, costs[ i ], PO );

			LastOpt = opt;

			if( opt -> getBestCost() <= BestOpt -> getBestCost() )
			{
				BestOpt = opt;
			}

			if( OptCount > 1 )
			{
				if( sc2 == 0 )
				{
					ATStallCount = 0;
				}
				else
				{
					ATStallCount++;
					CurOpt = PushOpt;

					if( OptCount == 2 )
					{
						PushOpt = Opts[ CurOpt == Opts[ 0 ]];
					}
					else
					{
						while( true )
						{
							PushOpt = Opts[ rnd.getInt( OptCount )];

							if( PushOpt != CurOpt )
							{
								break;
							}
						}
					}
				}
			}
			else
			{
				ATStallCount = sc2;
			}
		}

		CurBatchSize = 0;
	}

	/**
	 * @return Pointer to the i-th candidate vector from the most recent ask().
	 */

	const double* getAskValues( const int i ) const
	{
		return( Opts[ CandOptIdx[ i ]] -> getAskValues( i ));
	}

	/**
	 * Copy all candidate parameter vectors into a flat row-major buffer.
	 *
	 * @param[out] outBuf Output buffer, size CurBatchSize * ParamCount.
	 */

	void getAskValuesBuf( double* outBuf ) const
	{
		for( int i = 0; i < CurBatchSize; i++ )
		{
			memcpy( outBuf + i * ATParamCount, getAskValues( i ),
				ATParamCount * sizeof( double ));
		}
	}

	int getStallCount() const { return( ATStallCount ); }
	int getParamCount() const { return( ATParamCount ); }
	int getCurBatchSize() const { return( CurBatchSize ); }
	bool isInitPhase() const
	{
		return( CurOpt != NULL && CurOpt -> isInitPhase() );
	}

	virtual void getMinValues( double* const p ) const
	{
		memcpy( p, LowerBounds, ATParamCount * sizeof( double ));
	}

	virtual void getMaxValues( double* const p ) const
	{
		memcpy( p, UpperBounds, ATParamCount * sizeof( double ));
	}

	virtual double optcost( const double* const p )
	{
		(void) p;
		return( 1e300 );
	}

protected:
	class CBiteOptWrapAT : public CBiteOptAT
	{
	public:
		CBiteOptDeepAT* Owner;

		CBiteOptWrapAT( CBiteOptDeepAT* const aOwner )
			: Owner( aOwner )
		{
		}

		virtual void getMinValues( double* const p ) const
		{
			Owner -> getMinValues( p );
		}

		virtual void getMaxValues( double* const p ) const
		{
			Owner -> getMaxValues( p );
		}

		virtual double optcost( const double* const p )
		{
			return( Owner -> optcost( p ));
		}
	};

	int ATParamCount;
	int OptCount;
	CBiteOptWrapAT** Opts;
	CBiteOptWrapAT* BestOpt;
	CBiteOptWrapAT* CurOpt;
	CBiteOptWrapAT* PushOpt;
	CBiteOptWrapAT* LastOpt;
	int ATStallCount;

	double* LowerBounds;
	double* UpperBounds;

	int CurBatchSize;
	int* CandOptIdx;
	int* SortOrder;
	double* SortCosts;
	int MaxBatch;

private:
	static double sanitizeCost( const double cost )
	{
		return( cost == cost ? cost : 1e300 );
	}

	int getOptIndex( const CBiteOptWrapAT* const opt ) const
	{
		for( int i = 0; i < OptCount; i++ )
		{
			if( Opts[ i ] == opt )
			{
				return( i );
			}
		}

		return( 0 );
	}

	void ensureBatchBuffers( const int batchSize )
	{
		if( batchSize <= 0 || batchSize <= MaxBatch )
		{
			return;
		}

		MaxBatch = batchSize;

		for( int i = 0; i < OptCount; i++ )
		{
			Opts[ i ] -> initAskBuffers( MaxBatch );
		}

		delete[] CandOptIdx;
		delete[] SortOrder;
		delete[] SortCosts;

		CandOptIdx = new int[ MaxBatch ];
		SortOrder = new int[ MaxBatch ];
		SortCosts = new double[ MaxBatch ];
	}

	void deleteBuffers()
	{
		if( Opts != NULL )
		{
			for( int i = 0; i < OptCount; i++ )
			{
				delete Opts[ i ];
			}

			delete[] Opts;
			Opts = NULL;
		}

		delete[] LowerBounds;
		delete[] UpperBounds;
		delete[] CandOptIdx;
		delete[] SortOrder;
		delete[] SortCosts;

		LowerBounds = NULL;
		UpperBounds = NULL;
		CandOptIdx = NULL;
		SortOrder = NULL;
		SortCosts = NULL;
		BestOpt = NULL;
		CurOpt = NULL;
		PushOpt = NULL;
		LastOpt = NULL;
		ATParamCount = 0;
		OptCount = 0;
		ATStallCount = 0;
		CurBatchSize = 0;
		MaxBatch = 0;
	}
};

/**
 * Convenience function: minimization with ask/tell batching.
 * Drop-in replacement for biteopt_minimize() with an additional batchSize
 * parameter.
 */

inline int biteopt_minimize_batch( const int N, biteopt_func f, void* data,
	const double* lb, const double* ub, double* x, double* minf,
	const int iter, const int M = 1, const int attc = 10,
	const int batchSize = 8, const int stopc = 0,
	biteopt_rng rf = 0, void* rdata = 0, double* f_minp = 0 )
{
	const int ActualBatchSize = ( batchSize > 0 ? batchSize : 1 );

	CBiteOptDeepAT opt;
	opt.updateDims( N, M );

	CBiteRnd rnd;
	rnd.init( 1, rf, rdata );

	const int sct = ( stopc <= 0 ? 0 : 128 * N * stopc );
	const int useiter = (int) ( iter * sqrt( (double) M ));
	int evals = 0;
	double* costs = new double[ ActualBatchSize ];

	for( int k = 0; k < attc; k++ )
	{
		opt.init( rnd, lb, ub );

		bool IsFinished = false;
		int i = 0;

		while( i < useiter )
		{
			const int remaining = useiter - i;
			const int bs = ( remaining < ActualBatchSize ?
				remaining : ActualBatchSize );
			const int n = opt.ask( rnd, bs );

			if( n <= 0 )
			{
				break;
			}

			for( int j = 0; j < n; j++ )
			{
				costs[ j ] = f( N, opt.getAskValues( j ), data );
			}

			opt.tell( rnd, costs );

			i += n;
			evals += n;

			if( f_minp != 0 && opt.getBestCost() <= *f_minp )
			{
				IsFinished = true;
				break;
			}

			if( sct > 0 && opt.getStallCount() >= sct )
			{
				break;
			}
		}

		if( k == 0 || opt.getBestCost() <= *minf )
		{
			memcpy( x, opt.getBestParams(), N * sizeof( x[ 0 ]));
			*minf = opt.getBestCost();
		}

		if( IsFinished )
		{
			break;
		}
	}

	delete[] costs;

	return( evals );
}

#endif // BITEOPT_ASKTELL_INCLUDED
