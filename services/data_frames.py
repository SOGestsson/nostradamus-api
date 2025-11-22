import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import sys
import os


# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import pandas as pd
from crud.read import RioItemsRead, RioItemEditblesRead, HistoriesReadOneItem, OnOrderRead, HistoriesReadAllItems


class DataFrameFactory:
    def __init__(self, *, product_groups=None, location='Hafnarfjörður', min_moves=100, session=None):
        self.product_groups = product_groups
        self.location = location
        self.min_moves = min_moves
        self.session = session  # optional: share a session

    def create_rio_items_df(self):
        rio_items_reader = RioItemsRead(
            product_groups=self.product_groups, location=self.location, min_moves=self.min_moves, session=self.session
        )
        rio_item_editbles_reader = RioItemEditblesRead(
            product_groups=self.product_groups, location=self.location, min_moves=self.min_moves, session=self.session
        )

        rio_items = rio_items_reader.get_filtered_items()
        rio_item_editbles = rio_item_editbles_reader.get_filtered_items()

        rio_items_df = pd.DataFrame([i.__dict__ for i in rio_items]).drop(columns=['_sa_instance_state'], errors='ignore')
        rio_item_editbles_df = pd.DataFrame([i.__dict__ for i in rio_item_editbles]).drop(columns=['_sa_instance_state'], errors='ignore')

        combined_df = pd.merge(rio_items_df, rio_item_editbles_df, left_on='id', right_on='item_id', how='inner')
        combined_df.rename(columns={'id_x': 'id'}, inplace=True)
        combined_df.drop(columns=['id_y'], inplace=True)

        # typing / cleaning as you had:
        combined_df['buy_freq'] = pd.to_numeric(combined_df['buy_freq'], errors='coerce').astype('Int64')
        combined_df['vendor_name'] = combined_df['vendor_name'].fillna('UNKNOWN_CODE').astype(str)
        combined_df['vendor_country'] = combined_df['vendor_country'].fillna('UNKNOWN_COUNTRY').astype(str)
        combined_df['buy_freq'] = combined_df['buy_freq'].fillna(7)
        combined_df['days_in_stock'] = combined_df['days_in_stock'].round(0)
        combined_df['stock_level'] = combined_df['stock_level'].astype(int)
        combined_df['days_in_stock_order'] = combined_df['days_in_stock_order'].round(0)
        combined_df['last_year_usage_cost_value'] = combined_df['last_year_usage_cost_value'].round(0)

        return combined_df

    def create_all_history_dataframe(self):
        # histories are already filtered at the DB level using the same product_group(s)
        hist = HistoriesReadAllItems(
            product_groups=self.product_groups, location=self.location, min_moves=self.min_moves, session=self.session
        )
        query_result = hist.get_all_items_history()
        if not query_result:
            return pd.DataFrame(columns=['item_id', 'consumption_date', 'qty'])

        result_dicts = []
        for row in query_result:
            d = row.__dict__.copy()
            d.pop('_sa_instance_state', None)
            result_dicts.append(d)

        df = pd.DataFrame(result_dicts)
        if 'consumption_date' in df.columns:
            df = df.sort_values(by=['item_id', 'consumption_date'], ascending=True)
        return df

    def create_on_order_dataframe(self):
        rio_on_order = OnOrderRead(location=self.location, session=self.session).get_all_orders()
        rio_on_order_df = pd.DataFrame([i.__dict__ for i in rio_on_order]).drop(columns=['_sa_instance_state'], errors='ignore')
        return rio_on_order_df

class InvSimDataFrameFactory():
    ''' This class uses the class Datafram factory to creat input dataframes for the inventory simulator'''
    def __init__(self, df_his, df_rio_items, df_rio_on_order):
        self.sim_input_his = self.create_sim_input_his(df_his)
        self.sim_rio_items = self.create_sim_rio_items(df_rio_items)
        self.sim_rio_item_details = self.create_sim_rio_item_details(df_rio_items)
        self.sim_rio_on_order = self.create_sim_rio_on_order(df_rio_on_order)
     
    def add_missing_dates_to_sim_input_his(self, sim_input_his):
        all_pn = sim_input_his['item_id'].unique()
        i = 0

        min_date = sim_input_his['day'].min()
        #max_date = sim_input_his['day'].max()
        max_date = datetime.today().strftime('%Y-%m-%d')

        idx = pd.date_range(min_date, max_date)

        new_sim_input_his = sim_input_his.iloc[:0, :].copy()
        for column in all_pn:
            print(column)
            time_series_for_pn = sim_input_his[sim_input_his['item_id'] == column]
            time_series_for_pn = time_series_for_pn.set_index('day')
            time_series_for_pn = time_series_for_pn.reindex(idx)

            time_series_for_pn['item_id'] = time_series_for_pn['item_id'].fillna(column)
            time_series_for_pn['actual_sale'] = time_series_for_pn['actual_sale'].fillna(0)

            frames = [new_sim_input_his, time_series_for_pn]

            new_sim_input_his = pd.concat(frames)

        new_sim_input_his['day'] = new_sim_input_his.index

        return new_sim_input_his
     
    def create_sim_input_his(self, df_his):
        
        sim_input_his = df_his[['item_id', 'consumption_date', 'qty']]
        sim_input_his = sim_input_his.rename(columns={'consumption_date': 'day', 'qty': 'actual_sale'})
        sim_input_his['day'] = pd.to_datetime(sim_input_his['day'])

        #make sure that there are no duplicates in item_id, day
        sim_input_his = sim_input_his.groupby(['item_id', 'day']).agg({"actual_sale":"sum"}).reset_index()
        

        sim_input_his = sim_input_his[['item_id', 'actual_sale', 'day']]


        sim_input_his = self.add_missing_dates_to_sim_input_his(sim_input_his)
        
        return sim_input_his

    def create_sim_rio_items(self, df_rio_items):
       
        sim_rio_items = df_rio_items

        sim_rio_items = sim_rio_items[['item_id', 'description', 'stock_level', 'location', 'del_time', 'buy_freq', 'purchasing_method', 'min', 'max']]

        sim_rio_items.rename(columns={'stock_level': 'actual_stock', 'item_id': 'pn', 'location': 'station'}, inplace=True)
        
        # Change the data types of 'min', 'max', and 'actual_stock' to integers, handling NaN by filling with 0
        sim_rio_items['min'] = sim_rio_items['min'].fillna(0).astype(int)
        sim_rio_items['max'] = sim_rio_items['max'].fillna(0).astype(int)
        sim_rio_items['actual_stock'] = sim_rio_items['actual_stock'].fillna(0).astype(int)

        # Add the 'Unnamed: 0' column with value 1 as the first column
        sim_rio_items.insert(0, 'Unnamed: 0', 1)
        sim_rio_items.insert(4, 'ideal_stock', 0)
        sim_rio_items['purchasing_method'] = 'low_sale'
        return sim_rio_items

    def create_sim_rio_item_details(self, df_rio_items):
        '''This dataframe is just a dummy dataframe to run the inventory simulator as it is in the code we use'''


        sim_rio_item_details = df_rio_items
        sim_rio_item_details = sim_rio_item_details[['id', 'vendor_name']]

        return sim_rio_item_details   

    def create_sim_rio_on_order(self, df_rio_on_order):
        sim_rio_on_order = df_rio_on_order
        sim_rio_on_order = sim_rio_on_order[['item_number', 'est_deliv_date', 'est_deliv_qty']]
        sim_rio_on_order.rename(columns={'item_number': 'pn'}, inplace=True)
        sim_rio_on_order = sim_rio_on_order.groupby(['pn', 'est_deliv_date']).sum().reset_index()

        return(sim_rio_on_order)


if __name__ == "__main__":
    
    df = DataFrameFactory()
    rio_items_df = df.create_rio_items_df()
    print("Rio_items DataFrame created with shape:", rio_items_df.shape)
    rio_history_df = df.create_all_history_dataframe()
    print("Rio_history DataFrame created with shape:", rio_history_df.shape)
    rio_on_order_df = df.create_on_order_dataframe()
    print("Rio_on_order DataFrame created with shape:", rio_on_order_df.shape)
    
    alda = InvSimDataFrameFactory(rio_history_df, rio_items_df, rio_on_order_df)
    print("Simulation input history DataFrame created with shape:", alda.sim_input_his.shape)

    print(alda.sim_input_his)