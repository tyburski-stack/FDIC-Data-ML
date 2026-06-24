
from fdicfs.sources import synthetic as src
from fdicfs.features import get_features_asof

def test_poisoning_T_does_not_change_features():
    # beat 1
    clean = get_features_asof(src.T) # src.T is the predicted day 

    # beat 2
    poisoned = get_features_asof(src.T, 
                                 balance_source = src.poisoned_balance_asof,
                                 schedule_source= src.poisoned_schedule_asof
                                 )
    
    # beat 3 (test)
    assert clean == poisoned


# prove the test has teeth
def test_poison_test_has_teeth():
    # Deliberately LEAKY builder that illegally reads T
    def leaky_features_asof(T, portfolio_id = src.P, *, 
                            balance_source = src.get_balance_asof,
                            schedule_source = src.get_schedule_asof):
        balances = balance_source(T, portfolio_id) # reads T, not T-1
        last = balances.iloc[-1]
        return {"mmkt_balance": float(last[src.MMKT_BALANCE])}
    
    clean = leaky_features_asof(src.T)
    poisoned = leaky_features_asof(src.T, balance_source=src.poisoned_balance_asof)

    assert clean != poisoned