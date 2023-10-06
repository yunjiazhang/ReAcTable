
# CODEX + COTexe + WikiTQ
# jupyter nbconvert --to script COT-SQL-WikiTQ.ipynb
# /usr/bin/python3 COT-SQL-WikiTQ.py &> ./exp_logs/COT-SQL-WikiTQ.log

jupyter nbconvert --to script COT-SQL-WikiTQ-LeverVote.ipynb
/usr/bin/python3 COT-SQL-WikiTQ-LeverVote.py &> ./exp_logs/COT-SQL-WikiTQ-LeverVote.log

jupyter nbconvert --to script COT-MajorityVote-SQL-WikiTQ.ipynb
/usr/bin/python3 COT-MajorityVote-SQL-WikiTQ.py &> ./exp_logs/COT-MajorityVote-SQL-WikiTQ.log

# CODEX +  COTexe + TabFact
jupyter nbconvert --to script COT-SQL-TabFact.ipynb
/usr/bin/python3 COT-SQL-TabFact.py &> ./exp_logs/COT-SQL-TabFact.log

jupyter nbconvert --to script COT-SQL-TabFact-MajorityVote.ipynb
/usr/bin/python3 COT-SQL-TabFact-MajorityVote.py &> ./exp_logs/COT-SQL-TabFact-MajorityVote.log

jupyter nbconvert --to script COT-SQL-TabFact-LeverVote.ipynb
/usr/bin/python3 COT-SQL-TabFact-LeverVote.py &> ./exp_logs/COT-SQL-TabFact-LeverVote.log