'''
Runs the gSpan algorithm to find frequent subgraphs
'''

import os
import sys

from cStringIO import StringIO
old_stdout = sys.stdout
redirected_output = sys.stdout = StringIO()

sys.path.insert(0, '/home/dariog/repos/gSpan1')
data_path = '../exp3_subgraph_generator'
for file_name in os.listdir(data_path):
    sys.argv = ['/home/dariog/repos/gSpan1/main.py',os.path.join(data_path,file_name), '-l 3','-s 5000']
    #sys.argv = ['/home/dariog/repos/gSpan1/main.py',os.path.join(data_path,file_name)]
    try:
        execfile( "/home/dariog/repos/gSpan1/main.py")
    except:
        raise
    finally:
        sys.stdout = old_stdout
    with open('../exp4_fsd/frequent_subgraphs_'+file_name,'w') as f:
        f.write(redirected_output.getvalue())
    f.close()
