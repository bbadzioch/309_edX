# # XML Schema

# <bibliography xmlns="http://something.something.my.schema" version="0.1" >
# 
#         <item type="journal article">  
#                 <autor>John Smith</author>
#                 <autor>James Davis</author>
#                 <title>Methodology of science</title>
#                 <journal>Scientific letters</journal>
#                 <volume>14</volume>
#                 <pages>1034-1040</pages>
#                 <year>2012</year>
#         </item>
#         
#         <item type="book">  
#                 <autor>Mary Jones</author>
#                 <title>My life</title>
#                 <publisher>Jones Publishing Co.</publisher>
#                 <publisher>Jones Publishing Co.</publisher>
#                 <pages>345</pages>
#                 <year>1890</year>
#         </item>
# 
# 
#         <item type="thesis">  
#                 <autor>Vincent Vega</author>
#                 <title>History of Florence years 1200-1300</title>
#                 <college>University of Great Science</college>
#                 <pages>127</pages>
#                 <year>1890</year>
#         </item>
# 
#         </course>
# 
# </bibliography>

#50
import xmlschema

try:
    xmlschema.validate('mydoc.xml','myschema.xsd')
except xmlschema.XMLSchemaValidationError as e:
    print(e.obj, e.reason)
    

# # Regular expressions

#58
import re

re.findall(r'(\w)at', "the cat sat on the mat")

#3
import json


def format_cell(c):
    if c['cell_type'] == 'code':
        return "#" + str(c['execution_count']) + "\n" + ''.join(c['source'])
    if c['cell_type'] == 'markdown':
        return "# ".join(c['source'])

def get_notebook(nb_file, code_file=None):
    """
    Retrieves Python code from a Jupyter notebook file
    
    :nb_file:
        The name of an ipynb file
    :code_file:
        The name of the file where the code is to be saved. 
        If None nothing will be saved
    Returns:
        A string with the code
    """
    with open(nb_file) as foo:
        notebook = json.load(foo)
    cells = notebook['cells']
    
    cell_str = [format_cell(c) for c in cells] 
    s = '\n\n'.join(cell_str)
    if code_file is not None:
        with open(code_file, 'w') as foo:
            foo.write(s)
    return s

#9
print(get_notebook("mth448_week5.ipynb", code_file='sample.py'))

#None
