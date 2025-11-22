import pandas as pd
from datetime import datetime
try:
    from . import inventory_opt_and_forecasting_package as sim
except ImportError:
    import inventory_opt_and_forecasting_package as sim

import services.data_frames as df



    



class inv_opt_container(sim.inventory_simulator_with_input_prep):
    def __init__(self, df_his, df_rio_items, df_rio_on_order, number_of_days, number_of_simulations, service_level):

        self.final_res = self.run_purch_sugg_for_all_pn(df_his, df_rio_items, df_rio_on_order, number_of_days, number_of_simulations, service_level)
        self.purchase_suggestions = self.get_purchase_suggestion(self.final_res)

    def run_purch_sugg_for_all_pn(self, df_his, df_rio_items, df_rio_on_order, number_of_days, number_of_simulations, service_level):
        final_res = pd.DataFrame()
        


        # Iterate through all rows
        for index, row in df_rio_items.iterrows():
                item_id = row['item_id']
                print(item_id)
                print("------------------------------")
                print("------------------------------")
                print("------------------------------")
                filtered_df_his = df_his[df_his['item_id'] == item_id].copy()
                filtered_df_rio_items = df_rio_items[df_rio_items['item_id'] == item_id].copy()
                filtered_df_rio_on_order = df_rio_on_order[df_rio_on_order['item_id'] == item_id].copy()
        
                try:
                        # Your processing logic here
                        all_inputs = df.InvSimDataFrameFactory(filtered_df_his, filtered_df_rio_items, filtered_df_rio_on_order)
                except Exception as e:
                        print(f"Error processing item_id {item_id}: {e}")
                        # Optionally, log the error or add more error handling if needed
            
                sim_input_his = all_inputs.sim_input_his
                sim_rio_items = all_inputs.sim_rio_items
                sim_rio_item_details = all_inputs.sim_rio_item_details
                sim_rio_on_order = all_inputs.sim_rio_on_order

                sim_object = sim.inventory_simulator_with_input_prep(sim_input_his, sim_rio_items, sim_rio_on_order, sim_rio_item_details, number_of_days, number_of_simulations, service_level)

                if not final_res.empty:
                        final_res = pd.concat([final_res, sim_object.sim_result], ignore_index=True)
                       
                else:
                       final_res = sim_object.sim_result
                       


        return final_res

    def get_purchase_suggestion(self, final_res):
        # Find the rows with the oldest 'sim_date' for each 'item_id'
        oldest_dates = final_res.loc[final_res.groupby('item_id')['sim_date'].idxmin()]

        # Create a new DataFrame with 'item_id', 'purchase_qty', and current date and time
        purchase_suggestions = oldest_dates[['item_id', 'purchase_qty']].copy()
        purchase_suggestions['current_datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Display the new DataFrame
        return purchase_suggestions



if __name__ == '__main__':
      pass



  
   





  


