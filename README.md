# Solar MultiModel

Aplicação Streamlit para estimar a potência fotovoltaica de uma embarcação por
três modelos de complexidade crescente, executados sobre a mesma entrada e o
mesmo datasheet.

O software foi estruturado para manter a produção de uma estimativa mesmo em
modo degradado: se a temperatura ambiente deixa de chegar no CSV, o modelo de
irradiância continua operando durante toda a janela selecionada. Os modelos
térmico e SDM são marcados como indisponíveis, sem criar valores artificiais.

## Modelos implementados

### 1. Modelo de irradiância

Usa somente a irradiância efetiva e a potência nominal do datasheet:

```text
P = P_STC · G_ef / 1000
```

É o modelo de continuidade operacional. Não calcula temperatura de célula e
mantém a eficiência STC constante.

### 2. Modelo NOCT + eficiência

Calcula primeiro a temperatura da célula pelo NOCT:

```text
Tc = Tamb + (NOCT - 20) / 800 · G_ef
```

Depois corrige a eficiência pelo coeficiente de potência do datasheet:

```text
eta(Tc) = eta_STC · [1 + gamma_Pmax · (Tc - 25)]
P = eta(Tc) · G_ef · Area
```

### 3. Single Diode Model — SDM

Resolve a equação física completa do circuito equivalente de um diodo:

```text
I = IL - I0·[exp((V + I·Rs)/a) - 1] - (V + I·Rs)/Rsh
```

Os cinco parâmetros são extraídos do datasheet, transladados para cada par
`(G, Tc)` e usados para resolver o MPP minuto a minuto. O algoritmo conserva o
modelo original do repositório SolarSDM: Lambert W, verificação numérica e MPP
ideal.

Mais detalhes estão em [DOCUMENTACAO_MODELOS.md](DOCUMENTACAO_MODELOS.md).

## Interface

A aplicação possui cinco seções, acessadas por botões próprios no sidebar:

1. **Visão geral** — explicação da conversão fotovoltaica, das equações dos
   três modelos, do fluxo paralelo e da lógica de confiabilidade.
2. **Entrada** — botão `RODAR MODELOS` no topo, seleção do módulo e do arranjo,
   upload de CSV ou perfil sintético de 120 minutos/24 horas, curva solar
   perfeita sem ruído e validação da janela.
3. **Modelos** — três abas independentes com KPIs, potência, energia,
   comportamento térmico e eficiência em uma grade compacta. As curvas I-V/P-V
   e os indicadores elétricos do SDM ficam disponíveis sob demanda no painel
   de diagnóstico avançado.
4. **Comparação** — sobreposição de potência, energia e eficiência, diferença
   relativa ao SDM em uma grade 2 × 2 e tabela comparativa expansível.
5. **Exportação** — seleção do modelo, das colunas, do separador e do formato
   decimal antes de baixar o CSV.

## Contrato da entrada

Para o CSV, cada execução mantém o contrato operacional original:

- 120 linhas;
- passo temporal de 1 minuto;
- intervalo total de 2 horas;
- `timestamp` obrigatório;
- irradiância obrigatória;
- temperatura ambiente opcional.

O perfil sintético usa o mesmo passo de 1 minuto e permite escolher entre
120 minutos (2 horas) e 1.440 minutos (24 horas). A condição **Irradiância
perfeita** produz a envoltória solar diária suave, sem ruído ou quedas abruptas.

Exemplo com temperatura:

```csv
timestamp;GHI;Tamb
2026-03-21 12:01:00;976.8;29.9
2026-03-21 12:02:00;984.0;30.0
```

Exemplo degradado:

```csv
timestamp;GHI
2026-03-21 12:01:00;976.8
2026-03-21 12:02:00;984.0
```

O carregador aceita separador por vírgula ou ponto e vírgula, ponto ou vírgula
decimal e diferentes nomes usuais para timestamp, GHI e temperatura. Quando o
arquivo contém mais de 120 registros, a interface permite escolher o início da
janela.

A pasta `Dados_exemplo/` inclui quatro janelas completas e
`PREVISAO_SOLAR_120min_SEM_TEMPERATURA.csv` para testar o modo degradado.

## Módulos fotovoltaicos

A base mantém os 16 módulos Canadian Solar do SolarSDM original:

- CS6P, CS6K, CS6X e CS6U policristalinos;
- CS3W HiKu mono-PERC;
- CS7L HiKu7 mono-PERC, incluindo o **CS7L-580MS**.

O padrão operacional é:

- módulo: `CS7L-580MS`;
- arranjo: `2 módulos em série × 3 strings em paralelo`;
- 6 módulos no total;
- potência instalada: `3,480 kWp`;
- perdas ópticas: `0 %`.

Os modelos 1 e 2 recebem automaticamente `Pnom`, área, NOCT, eficiência e
`gamma_Pmax` do módulo selecionado. O SDM recebe também os parâmetros elétricos
e os cinco parâmetros extraídos.

## Instalação local

Requer Python 3.10 a 3.12.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instale e execute:

```bash
pip install -r requirements.txt
streamlit run app.py
```

A aplicação abrirá normalmente em `http://localhost:8501`.

## Streamlit Community Cloud

1. Envie todo o conteúdo desta pasta para o repositório GitHub.
2. Crie ou atualize o aplicativo no Streamlit Community Cloud.
3. Selecione `app.py` como arquivo principal.
4. Não é necessário configurar comandos adicionais.

## Testes

```bash
python -m unittest discover -s tests -v
```

Os testes verificam:

- convergência da extração SDM;
- potência nominal do modelo simples em irradiância STC;
- derating térmico do modelo NOCT;
- execução SDM da janela completa;
- geração sintética de 24 horas e suavidade da irradiância perfeita;
- modo degradado sem temperatura;
- contrato do CSV;
- consistência dos KPIs e da exportação;
- inicialização da interface Streamlit.

## Estrutura

```text
solarsdm-multimodel/
├── app.py
├── requirements.txt
├── DOCUMENTACAO_MODELOS.md
├── assets/
│   └── fluxo_fotovoltaico.jpg
├── Dados_exemplo/
├── config/
│   ├── pv_database.py
│   └── settings.py
├── models/
│   ├── irradiance_model.py
│   ├── pv_module.py
│   ├── single_diode.py
│   └── temperature_model.py
├── simulation/
│   ├── multimodel.py
│   ├── mpp.py
│   └── solver.py
├── visualization/
│   └── multimodel_plots.py
└── tests/
    ├── test_multimodel.py
    └── test_app_smoke.py
```

## Observações científicas

- Os três modelos aumentam a disponibilidade e permitem detectar divergências;
  isso não substitui validação contra potência medida.
- Os três compartilham a mesma irradiância, portanto uma falha nessa variável é
  uma falha de modo comum.
- A irradiância selecionada é interpretada como irradiância incidente no plano
  do módulo.
- O SDM assume MPPT ideal. Inversor, bateria, sombreamento parcial e dinâmica do
  conversor não fazem parte desta versão.

Antes do depósito definitivo, complete no repositório os autores, a licença e
a forma de citação acordadas pela equipe do projeto.
