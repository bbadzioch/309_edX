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

# In[50]:
import xmlschema

try:
    xmlschema.validate('mydoc.xml','myschema.xsd')
except xmlschema.XMLSchemaValidationError as e:
    print(e.obj, e.reason)
    

# # Regular expressions

# In[58]:
import re

re.findall(r'(\w)at', "the cat sat on the mat")

# # JSON

# In[21]:
import json




def get_notebook(nb_file, save=False):
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
    
    def format_cell(c):
        if c['cell_type'] == 'code':
            return "#" + str(c['execution_count']) + "\n" + ''.join(c['source'])
        if c['cell_type'] == 'markdown':
            return "# " + "# ".join(c['source'])
    
    with open(nb_file) as foo:
        notebook = json.load(foo)
    cells = notebook['cells']
    
    cell_str = [format_cell(c) for c in cells] 
    s = '\n\n'.join(cell_str)
    if save:
        fname = nb_file[:nb_file.find(".ipynb")]
        with open(fname + '.py', 'w') as foo:
            foo.write(s)
    return s

# In[24]:
get_notebook("mth448_week5.ipynb", save=True);

# # NHTSA complaints database

# In[38]:
import requests
import json
url0 = 'http://www.nhtsa.gov/webapi/api/Complaints/vehicle/modelyear/{}/make/{}/model/{}?format=json'
year,make,model = '2009','Chevrolet','Cobalt'
url = url0.format(year,make,model)
s = requests.get(url).text  # a JSON string
complaints = json.loads(s)

# In[39]:
complaints

# In[69]:
year_min = 1980
year_max = 2018
make = 'Hyundai'
model = 'Sonata'

def count_complaints(make, model, year_min, year_max):
    complaint_count = {}
    for year in range(year_min, year_max + 1):
        url0 = 'http://www.nhtsa.gov/webapi/api/Complaints/vehicle/modelyear/{}/make/{}/model/{}?format=json'
        url = url0.format(year,make,model)
        s = requests.get(url).text  # a JSON string
        complaints = json.loads(s)
        if complaints['Message'] == 'No results found for this request':
            count = -1
        else:
            count = complaints['Count']
        complaint_count[year] = count
    
    return complaint_count

# In[70]:
import matplotlib.pyplot as plt

year_min = 1980
year_max = 2018
make = 'Hyundai'
model = 'Sonata'

count = count_complaints(make, model, year_min, year_max)
plt.plot(*zip(*count.items()))
plt.show()

# In[71]:
import matplotlib.pyplot as plt

year_min = 1980
year_max = 2018
make = 'Toyota'
model = 'Prius'

count = count_complaints(make, model, year_min, year_max)
plt.plot(*zip(*count.items()))
plt.show()

# In[None]:
