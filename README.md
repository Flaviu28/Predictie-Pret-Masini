# Proiect Sisteme Inteligente: Predictia pretului pentru masini rulate

## Descrierea temei
Proiectul va utiliza algoritmi de Machine Learning pentru a estima pretul de vanzare al masinilor rulate. Scopul este de a crea un sistem ce va prezice valoarea de piata a unui vehicul pe baza specificatiilor sale tehnice si a istoricului de utilizare.

## Structura setului de date (cardekho.csv)
Dataset-ul utilizat contine urmatoarele coloane:

* name : Marca si modelul autoturismului
* year : Anul de fabricatie
* selling_price : Pretul de vanzare
* km_driven : Numarul de kilometri parcursi
* fuel : Tipul de combustibil utilizat
* seller_type : Tipul vanzatorului (Individual/ Dealer/ Trustmark Dealer)
* transmission : Tipul transmisiei (Manual/ Automatic)
* owner : Istoricul de proprietate
* mileage(km/ltr/kg) : Consumul de combustibil
* engine : Capacitatea cilindrica a motorului
* max_power : Puterea maxima a motorului
* seats : Numarul de locuri

## 3. Metodologie si Preprocesare
Pentru a transforma datele brute intr-un set de date am parcurs urmatoarele etape:

* **Curatarea datelor (Data Cleaning)**: 
    * Am extras valorile numerice din coloanele `mileage`, `engine` si `max_power` folosind expresii regulate (Regex).
    * Am eliminat coloana `seats` (sau am tratat-o ca fiind optionala) pentru a ne concentra pe parametrii tehnici principali.
    * Valorile lipsa (NaN) au fost completate cu media coloanelor respective pentru a nu pierde date importante.
* **Eliminarea anomaliilor (Outliers)**: 
    * Am utilizat metoda **IQR (Interquartile Range)** pentru a detecta si elimina inregistrarile cu preturi de vanzare extreme (foarte mari sau nerealist de mici) care ar putea induce in eroare modelul de predictie.

## 4. Analiza Indicatorilor Inteligenti
In aceasta etapa am analizat modul in care informatia este distribuita si cum influenteaza variabilele pretul final:

* **Corelatia (Pearson)**: Am calculat matricea de corelatie pentru a vedea legatura liniara dintre variabile. S-a observat o corelatie pozitiva puternica intre `max_power` (puterea motorului) si `selling_price`.
* **Information Quantity (Mutual Information)**: Am utilizat acest indicator din teoria informatiei pentru a masura gradul de dependenta dintre caracteristici si pret. Spre deosebire de corelatie, acesta identifica si relatiile non-liniare.
* **Gini Index**: Am analizat conceptul de impuritate a datelor (Gini Impurity). Acest indicator ne ajuta sa intelegem cat de amestecate sunt preturile in setul de date.

## 5. Tehnologii Utilizate
* **Python**: Limbajul principal de programare.
* **Pandas**: Pentru manipularea tabelelor de date.
* **Scikit-Learn**: Pentru calcularea indicatorilor de informatie si preprocesare.
* **Seaborn & Matplotlib**: Pentru generarea matricelor de corelatie si a graficelor de analiza.
