from bs4 import BeautifulSoup

class Epicrise:
    def __init__(self, text: str):
        self.soup = BeautifulSoup(text, 'html.parser')

    def get_text(self):
        text : str = ''
        text += self.soup.find('code', code='COMPLNTS').parent.find('text').text
        text += self.soup.find('code', code='ANAM').parent.find('text').text
        # text += self.soup.find('code', code='PHYS').parent.find('entry').find('value').text.string
        for row in  self.soup.find('code', code='RESINSTR').parent.find_all('entry'):
            res = row.find('code')['displayname']
            info = row.find('value').text
            text += f'{res} - {info}'
        return text

    def get_disease(self):
        main_dis = self.soup.find('code', code='DGN').parent.find('entryrelationship').find('value')
        return main_dis['code']  + ' ' + main_dis['displayname']