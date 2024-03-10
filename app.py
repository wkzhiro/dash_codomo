from stl import mesh
import numpy as np
import pandas as pd
import base64
import io
import requests

from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from joblib import load

import dash
from dash import html,dcc,ClientsideFunction, no_update
from dash.dependencies import Input, Output, State
import dash_uploader as du
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objs as go


upload_style = {
    "width": "100%",
    "height": "120px",
    "lineHeight": "60px",
    "borderWidth": "1px",
    "borderStyle": "dashed",
    "borderRadius": "5px",
    "textAlign": "center",
    "margin": "10px",
    "margin": "3% auto",
}

def generate_table(dataframe, max_rows=30):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

def generate_tablefigure(dataframe, max_rows=50):
    # 表示する最大行数を制限
    if len(dataframe) > max_rows:
        dataframe = dataframe.iloc[:max_rows]

    # ヘッダー作成
    header_values = list(dataframe.columns)
    
    # セルのデータ作成
    cells_values = [dataframe[col].tolist() for col in dataframe.columns]
    
    # PlotlyのTableトレース作成
    trace = go.Table(
        header=dict(values=header_values, fill_color='paleturquoise', align='left'),
        cells=dict(values=cells_values, fill_color='lavender', align='left')
    )
    
    # フィギュアの作成と返却
    figure = go.Figure(data=[trace])
    
    return figure

#日付間のデータを表示
def generate_grouped_bar_chart(dataframe, start_date, end_date):
    # 日付範囲でデータフレームをフィルタリング
    filtered_df = dataframe.loc[start_date:end_date]

    # 積み上げ棒グラフのデータ生成
    data = []
    # 列の順番を決定
    column_order = ["sekkei", "seizou1", "seizou2", "seizou3", "seizou4", "kumitate"]
    # 列の順番に従ってデータを追加
    for col in column_order:
        for i, row in filtered_df.iterrows():
            # 各列のインデックスを取得し、オフセットを計算
            offset = column_order.index(col) * 0.1  # オフセットの値を調整することで列間の間隔を変更できます
            # X軸の値を日付とオフセットの組み合わせに設定
            x_value = i + pd.Timedelta(offset, 'D')
            # prediction の値に基づいて色を設定
            color = 'rgba(55, 128, 191, 0.7)' if row['prediction'] else 'rgba(219, 64, 82, 0.7)'
            # 棒グラフを追加
            data.append(go.Bar(
                x=[x_value],
                y=[row[col]],
                name=col,
                marker=dict(color=color)
            ))

    # レイアウト設定
    layout = go.Layout(
        title='Stacked Bar Chart',
        barmode='stack',  # 棒グラフを積み上げる
        xaxis={'title': 'Date', 'type': 'category'},
        yaxis={'title': '週キャパシティ比率'}
    )

    # フィギュアの作成と返却
    return go.Figure(data=data, layout=layout)

#特定の日付で表示
def generate_bar_chart_filter(dataframe, date):
    # 特定の日付のデータをフィルタリング
    specific_date = dataframe.loc[[date]]

    # 棒グラフのデータを生成
    data = [
        go.Bar(
            x=["sekkei", "seizou1", "seizou2", "seizou3", "seizou4","kumitate"],  # X軸にカテゴリ名
            y=specific_date.iloc[0][["sekkei", "seizou1", "seizou2", "seizou3", "seizou4","kumitate"]].values,  # Y軸に特定日付の各カテゴリの値
            marker_color='indianred'  # 棒の色
        )
    ]

    # レイアウト設定
    layout = go.Layout(
        title='製造予定',
        xaxis_title='division',
        yaxis_title='週キャパシティ比率',
        barmode='group'
    )

    return  go.Figure(data=data, layout=layout)

#stlデータからのモデル情報計算
def extract_features_from_stl(stl_path):
    # STLファイルの読み込み
    model_mesh = mesh.Mesh.from_file(stl_path)
    # # バイト列からメッシュを読み込む
    # model_mesh = mesh.Mesh.from_buffer(stl_bytes)
    # ボリュームと表面積を計算
    volume, _, _ = model_mesh.get_mass_properties()
    area = model_mesh.areas.sum()
    return volume, area

#stlデータからの予測
def stml_to_predict(file_path):

    # if contents is None:
    #     # contentsがNoneの場合は、処理をスキップまたは適切なデフォルト値を返します。
    #     return None
    # else:
    #     content_type, content_string = contents.split(",")
    #     decoded = base64.b64decode(content_string)

        try:
            # if filename.endswith(".stl"):
            # 保存されたモデルを読み込む
            loaded_model = load('model_multioutput.joblib')
            
            # STLファイルを扱うためにBytesIOオブジェクトを使用
            # stl_file = io.BytesIO(decoded)
            # 特徴量を抽出
            new_volume, new_area = extract_features_from_stl(file_path)
            # 新規データの特徴量データフレームの作成
            # ここでは、新規データに存在する特徴量のみを使用しています。
            new_data = pd.DataFrame({
                'Volume': [new_volume],  # 新規STLファイルから抽出したボリューム
                'Area': [new_area]       # 新規STLファイルから抽出した表面積
            })

            predictions = loaded_model.predict(new_data)

            return predictions[0]
        except Exception as e:
            print(e)
            return html.Div(["ファイルの読み込みでエラーが発生しました.stlデータを入力してください。"])
    
#日付から先週の月曜日を出力
def previous_monday(date):
    # 選択された日付を変換
    selected_date = pd.to_datetime(date)
    
    # 選択された日付の曜日を取得（月曜日が0、日曜日が6）
    weekday = selected_date.weekday()
    
    # 選択された日付からその週の月曜日（weekday日前）を引いた後、さらに7日前（1週間前）の日付を求める
    previous_monday_date = selected_date - pd.Timedelta(days=weekday + 7)
    
    return previous_monday_date

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, 
                external_stylesheets=external_stylesheets, 
                suppress_callback_exceptions=True,
                )

# 週の開始日（月曜日）のリストを生成
start_date = pd.to_datetime('2023-01-01')  # 開始日
end_date = pd.to_datetime('2025-12-31')  # 終了日
dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')

# ドロップダウンの選択肢
options = [{'label': i.strftime('%Y-%m-%d'), 'value': i} for i in dates]

du.configure_upload(app, "uploads/")  # ファイルをアップロードするディレクトリを設定

app.layout = html.Div([


    html.H4(children='BI tool Project'),


    # 月曜日の日付のドロップダウンリストと次の日曜日の日付を表示する領域を横並びに配置
    html.Div([
    # "納期日：" テキストの追加
    html.Label('納期日：', style={'margin-right': '10px'}),  # テキストとドロップダウンの間にマージンを追加

    dcc.DatePickerSingle(
        id='deadline',
        date='2024-01-01',
        display_format='YYYY-MM-DD',  # 日付の表示形式を 'YYYY-MM-DD' に設定
    ),
    ], style={'display': 'flex', 'align-items': 'center'}),  # 要素を横に並べ、中央揃えにする,
    
    html.Div([
        html.Label('表示範囲：', style={'margin-right': '10px','width': '10%'}),

        # 月曜日の日付のドロップダウンリスト
        dcc.Dropdown(
            id='monday-dropdown',
            options=options,
            value=dates[0].strftime('%Y-%m-%d'),  # デフォルト値はリストの最初の日付
            style={'width': '40%',}  # ドロップダウンの幅を調整
        ),
        html.H6(children="~", style={'margin-left': '10px', 'margin-right': '10px'}),
        # 次の日曜日の日付を表示する領域
        html.Div(id='3month_later_date', style={'width': '50%', 'paddingLeft': '5px'}),  # 左の要素との間隔を設定
    ], style={'display': 'flex', 'align-items': 'center','paddingBottom': '20px','flex-wrap': 'nowrap','justify-content': 'space-between'}),  # Flexboxを使って横並びに配置し、下部に余白を追加

    html.H6("Upload with Dash-uploader"),
            du.Upload(
                id="uploader",
                # children=html.Div(["stmlファイルをアップロードしてください。Drag and Drop Here to upload"]),
                # style=upload_style,
                filetypes=['stl'],
                max_files=1,
                cancel_button=True,
            ),

    html.H6(""),
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='新規受注予測', value='tab-1'),
        dcc.Tab(label='生産状況確認', value='tab-2'),
        dcc.Tab(label='受注', value='tab-3')
    ]),
    dcc.Graph(id="prediction_graph"),
    html.Div(id='tabs-example-content'),
    html.Button('受注', id='register_button', style={'display': 'none',},n_clicks=0),
    html.Div(id='response-result') , # レスポンスを表示するコンポーネント

    #結果の格納
    dcc.Store(id="prediction_data", storage_type="memory"),
    dcc.Store(id="total_data", storage_type="memory"),
    dcc.Store(id='response-data', storage_type='memory'),
    dcc.Store(id="current-sch", storage_type="memory")

])

# コールバックでドロップダウンの選択に応じて3か月後の末日を表示
@app.callback(
    Output('3month_later_date', 'children'),
    [Input('monday-dropdown', 'value')]
)
def update_sunday(selected_monday):
    # 選択された月曜日の3か月後を求める
    three_months_later  = pd.to_datetime(selected_monday) + pd.DateOffset(months=3)

    # 3か月後の月の最終日を求める
    last_day_of_three_months_later = three_months_later + pd.offsets.MonthEnd(0)

    return f"{last_day_of_three_months_later.strftime('%Y-%m-%d')}"

# 納期日に従って、1年後までのデータを取得
@app.callback(
    Output('current-sch', 'data'),
    Input('deadline', 'date')
)
def get_schedule(date):
    print("get_schedule")

    date_string = date
    date_format ="%Y-%m-%d"
    date = datetime.strptime(date_string, date_format)

    # relativedelta を使用して日付を計算
    start_date = date - relativedelta(months=6)
    end_date = date + relativedelta(months=6)

    # datetime オブジェクトを指定したフォーマットの文字列に変換
    start_date_string = start_date.strftime(date_format)
    end_date_string = end_date.strftime(date_format)

    # 最終的な日付データを文字列として格納
    date_data = {
        "start_date": start_date_string,
        "end_date": end_date_string
    }

    # # DateRangeの定義
    # date_data = {
    # "start_date": (date- relativedelta(months=6)),
    # "end_date": (date + relativedelta(months=6))
    # }

    try:    
        url = "http://127.0.0.1:8000/GraphData/"
        response = requests.post(url,json=date_data)
        data_schedule = response.json()
        print("data_schedule", data_schedule[1])
        df = pd.DataFrame(data_schedule)
        # print(df.head())
        df = df.drop(['results_Graph_id',"results_id",'status','quote_Graph_id','prediction_id', 'prediction_Graph_id'],axis=1)
        print(df.head(),df.columns)

        # dateカラムをdatetime型に変換
        df['date'] = pd.to_datetime(df['date'])

        # dateをインデックスに設定
        df.set_index('date', inplace=True)
        # Pivot table to get 'division_id' as columns
        current_df = df.pivot_table(values='man_hours', index=df.index, columns='division_id', aggfunc='first')
        current_df = current_df.fillna(0)

        # 必要なカラムだけを選択（存在しないdivision_idの場合はカラムが作成されないため、fillna(0)で補完）
        current_df['prediction'] = False

        current_df = current_df.reset_index()
        current_df_dict = current_df.to_dict('records')

        print(current_df_dict[0])

        return current_df_dict

    except requests.exceptions.JSONDecodeError:
        print("JSON データをデコードできませんでした。レスポンス: ", response.text)


#アップロードして計算
@app.callback(
        [
            Output("prediction_data", "data"),
            Output("total_data", "data"),
            Output("prediction_graph", "figure"),
            Output("monday-dropdown","value")
        ],
        [
            Input("uploader", "isCompleted"), 
            Input('tabs-example', 'value'), 
            Input("monday-dropdown","value")],
        [
            State("uploader", "fileNames"), 
            State("uploader", "upload_id"),
            State("deadline", "date"),
            State("monday-dropdown", "value"),
            State("3month_later_date","children"),
            State("prediction_data", "data"),
            State('total_data', 'data'),
            State("current-sch", "data")],
        prevent_initial_call=True,
    )
def stl_prediction(isCompleted,tab, start_date_ch,fileNames, upload_id, selected_date,start_date,end_date,data,total_data,schedule):
    # デフォルト値の設定
    default_data = None  # テーブルデータ用のデフォルト値（空リスト）
    default_data2 = []  # テーブルデータ用のデフォルト値（空リスト）
    default_figure = go.Figure()  # グラフ用のデフォルト値（空のFigure）

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # 選択された日付からその週の月曜日の日付を求める
    previous_monday_date = previous_monday(selected_date)

    #input1がトリガーとなった場合の処理
    print("data",data)

    if trigger_id == 'tabs-example':
        if tab == 'tab-1':
            print("tab",tab)
            # default_figure = go.Figure()  # グラフ用のデフォルト値（空のFigure）
            try:
                pd_prediction = pd.DataFrame(data)
                pd_prediction_pivot = pd_prediction.pivot_table(index='date', columns='division_id', values='man_hours', aggfunc='sum').fillna(0)
                pd_prediction_pivot = pd_prediction_pivot[pd_prediction_pivot.columns[::-1]]

                graph_figure = generate_tablefigure(pd_prediction) 
                
                return data,total_data,graph_figure,start_date
                
            except Exception as e:
                print("error")
                print(e)
                

            return default_data, default_data2, default_figure, start_date
        elif tab == 'tab-2':
            print("tab",tab)
            # default_figure = go.Figure()  # グラフ用のデフォルト値（空のFigure）
            try:
                schedule_df = pd.DataFrame(
                        total_data,
                        columns=["date","sekkei","seizou1","seizou2","seizou3","seizou4","kumitate","prediction"],
                    )
                print(schedule_df)
                # 各列を特定の値で割って、100を掛けてパーセンテージに変換
                schedule_df["sekkei"] = (schedule_df["sekkei"] / 48) * 100
                schedule_df["seizou1"] = (schedule_df["seizou1"] / 118) * 100
                schedule_df["seizou2"] = (schedule_df["seizou2"] / 368) * 100
                schedule_df["seizou3"] = (schedule_df["seizou3"] / 184) * 100
                schedule_df["seizou4"] = (schedule_df["seizou4"] / 168) * 100
                schedule_df["kumitate"] = (schedule_df["kumitate"] / 120) * 100

                schedule_df['date'] = pd.to_datetime(schedule_df['date'])
                schedule_df = schedule_df.set_index("date")
                # print(schedule_df)
                # print(start_date, end_date)
                bar_schedule_graph = generate_grouped_bar_chart(schedule_df, start_date,end_date)
                
                return data,total_data,bar_schedule_graph,start_date
                
            except Exception as e:
                print("error")
                print(e)
                
            return default_data, default_data2, default_figure, start_date
    
    elif trigger_id == 'uploader':
        print(fileNames, upload_id)

        if isCompleted and fileNames:
            try:
                stl_path = f"uploads/{upload_id}/{fileNames[0]}"  # アップロードされたファイルのパス
                print("selected",selected_date)
                selected_date = selected_date.replace("-","")
                url = f"http://127.0.0.1:8000/stlmodel/{selected_date}" #パスパラメーターで納品日を記載
                file = {'upload_file': open(stl_path, 'rb')} #”upload_file”としてファイルを定義
                
                response = requests.post(url=url, files=file)
                model_graph = response.json()
                pd_prediction = pd.DataFrame(model_graph)
                print("pd_prediction",pd_prediction.head())

                #ここでFastAPIにPOSTするが、代わりにcsvからの結果を取得
                # pd_prediction = pd.read_csv("model_Graph.csv")
                # 日付列をdatetime型に変換
                pd_prediction['date'] = pd.to_datetime(pd_prediction['date'])
                # division_idを列にピボットし、man_hoursを値として使用
                pd_prediction_pivot = pd_prediction.pivot_table(index='date', columns='division_id', values='man_hours', aggfunc='sum').fillna(0)
                pd_prediction_pivot = pd_prediction_pivot[pd_prediction_pivot.columns[::-1]]
                pd_prediction_pivot["prediction"] = True
                process_start_date = pd_prediction_pivot.index[0]
                # prediction = stml_to_predict(stl_path)
                # index = [selected_date]
                # pd_prediction = pd.DataFrame(
                #         [prediction], 
                #         columns=["sekkei","seizou1","seizou2","seizou3","seizou4","kumitate", '時間合計', '見積金額', '決定金額'],
                #         index = pd.to_datetime(index)
                #     )
                # pd_prediction_process = pd_prediction[["sekkei","seizou1","seizou2","seizou3","seizou4","kumitate"]]
                # pd_prediction_process["confirmed"] = True

                # 新しいデータフレーム df2 を既存のデータフレーム final_df に結合
                # ここでは concat を使用し、axis=0 で行方向に結合
                current_df = pd.DataFrame(schedule)
                print("current_df",current_df)
                current_df['date'] = pd.to_datetime(current_df['date'])
                current_df = current_df.set_index("date")

                total_df = pd.concat([current_df, pd_prediction_pivot], axis=0).sort_index()

                # # 同じ週のデータがある場合、値を合計する
                # # groupbyとsumを使用してインデックス（週の開始日）でグループ化し、値を合計する
                # # 予測値と確定値を合計したdf
                # total_df = total_df.groupby(total_df.index).sum()
                total_df = total_df.reset_index()
                # total_df['prediction'] = total_df['prediction'].astype(int).astype(bool)

                total_df_dict = total_df.to_dict('records')

                #予測値の保存
                pd_prediction_dict = pd_prediction.to_dict('records')

                #予測結果のグラフ作成
                graph_figure = generate_tablefigure(pd_prediction) 
                return pd_prediction_dict, total_df_dict, graph_figure, process_start_date
            
            except Exception as e:
                print(e)
    elif trigger_id == 'monday-dropdown':
        print("tab",tab)
        # default_figure = go.Figure()  # グラフ用のデフォルト値（空のFigure）
        try:
            schedule_df = pd.DataFrame(
                    total_data,
                    columns=["date","sekkei","seizou1","seizou2","seizou3","seizou4","kumitate","prediction"],
                )
            print(schedule_df)
            # 各列を特定の値で割って、100を掛けてパーセンテージに変換
            schedule_df["sekkei"] = (schedule_df["sekkei"] / 48) * 100
            schedule_df["seizou1"] = (schedule_df["seizou1"] / 118) * 100
            schedule_df["seizou2"] = (schedule_df["seizou2"] / 368) * 100
            schedule_df["seizou3"] = (schedule_df["seizou3"] / 184) * 100
            schedule_df["seizou4"] = (schedule_df["seizou4"] / 168) * 100
            schedule_df["kumitate"] = (schedule_df["kumitate"] / 120) * 100

            schedule_df['date'] = pd.to_datetime(schedule_df['date'])
            schedule_df = schedule_df.set_index("date")
            # print(schedule_df)
            # print(start_date, end_date)
            bar_schedule_graph = generate_grouped_bar_chart(schedule_df, start_date_ch,end_date)
            
            return data,total_data,bar_schedule_graph,start_date_ch
            
        except Exception as e:
            print("error")
            print(e)
            
        return default_data, default_data2, default_figure, start_date_ch


    # すべての実行パスの最後でデフォルト値を返す
    return default_data, default_data2, default_figure, start_date



#登録ボタンの表示
@app.callback(
    [
        Output('register_button', 'disabled'),# 登録ボタンのdisabled属性を制御
        Output("register_button","style")
    ],  
    Input('tabs-example', 'value'),  # タブの値
    State('prediction_data', 'data'),  # 特定のデータ入力フィールドからの入力値
)
def toggle_register_button(tab, data):
    if tab == 'tab-2':  # タブがtab3の場合
        if data==None:
            print(1,data)
            return True,{ # ボタンを活性化（disabled=False）
                'display': 'block',
                'backgroundColor': 'grey',  # 非活性化状態の色
                'color': 'white',
                'border': '1px solid grey'
                }  
        else:
            print(2)
            return False,{
                'display': 'block',
                'backgroundColor': 'green',  # 活性化状態の色
                'color': 'white',
                'border': '1px solid green',
                'opacity': '0.5'
                }
    else:
        print(3)
        return True, {'display': 'none'}   # それ以外の場合はボタンを非活性化（disabled=True）


#登録ボタンの操作時
@app.callback(
    Output('response-result', 'children'),
    [Input("register_button", "n_clicks")],
    State("prediction_data","data")
)
def send_request(n_clicks,data):
    if n_clicks is None or n_clicks == 0:
        print("click-None")
        return None
    print("click!")

    df = pd.DataFrame(data)

    # date列をdatetime型に変換
    df['date'] = pd.to_datetime(df['date'])

    # 'division_id'に基づいて'man_hours'をpivot
    df_pivot = df.pivot_table(
        index=['cavity_number'],
        columns='division_id',
        values='man_hours',
        aggfunc='sum',
        fill_value=0  # NaNを0で置き換える
    ).reset_index()

    # 各グループの最初と最後の日付を計算
    print("df_pivot", df_pivot)
    result  = df.groupby(['cavity_number'])['date'].min()
    start_date = result.reset_index()['date'][0]
    result  = df.groupby(['cavity_number'])['date'].max()
    end_date = result.reset_index()['date'][0]

    # datetime.datetime 型から datetime.date 型に変換
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    print(start_date,end_date)

    df_pivot['start_day'] = start_date
    df_pivot['end_day'] = end_date
    first_row = df_pivot.iloc[0]
    json_data = first_row.to_dict()
    
    # date 型のデータを ISO 形式の文字列に変換
    for key, value in json_data.items():
        if isinstance(value, datetime):
            json_data[key] = value.isoformat()
        elif isinstance(value, date):
            json_data[key] = value.isoformat()


    print(json_data)
    # ここでリクエストを送信
    url = "http://127.0.0.1:8000/writeData"  # 例としてのURL
    response = requests.post(url, json=json_data)  # POSTリクエストを送信

    # サーバーからのレスポンスを表示
    return f"サーバーからのレスポンス: {response}"

#モーダルの表示、うまく行かないため不使用
@app.callback(
    [Output('register-modal', 'is_open'), Output('modal-body', 'children')],
    [Input('response-data', 'data')]  # クライアントサイドからのレスポンスデータ
)
def update_modal(response_data):
    print("response",response_data)
    if response_data:
        return True, response_data  # モーダルを開き、レスポンスデータを表示
    return False, ""  # レスポンスデータがない場合はモーダルを閉じる

if __name__ == '__main__':
    app.run_server(debug=True, )