import sys
from nltk import edit_distance

intents = ['ListObject', 'Get_bill', 'Add_item_to_list', 'Get_list', 'Record_video', 'GetGenericBusinessType', 'Initiate_call', 'Call_parameter_setting', 'Get_product', 'Get_health_stats', 'Organization_contact', 'Log_nutrition', 'Phone_number', 'Create_list', 'Personal_contact', 'Get_note', 'Create_note', 'Local_business_contact', 'Pause_exercise', 'Get_security_price', 'BuyEventTickets', 'Play_game', 'ListItemObject', 'Open_app', 'Resume_exercise', 'Place', 'Take_photo', 'Email_address', 'Find_parking', 'Cancel_ride', 'ArgumentFactoryDatetimeDurationRecurrence', 'CheckableString', 'Post_message', 'RD', 'Screenshot', 'Electronic_attachment', 'CulinaryQuantity', 'List_position', 'Contactable_entity', 'MenuItemQuantity', 'Family_contact', 'Electronic_message', 'Check_order_status', 'Person_location', 'Stop_exercise', 'Get_message_content', 'Send_digital_object', 'Other', 'ActivatedGivenness', 'Add_contact', 'Start_exercise', 'Pay_bill', 'AllOf', 'Order_menu_item', 'Emergency_number_contact', 'NonNegativeSimpleNumber', 'Order_ride', 'Log_exercise']
slots = ['capture_mode', 'note_feature', 'account_type', 'value', 'text', 'object', 'order_fulfillment', 'contact', 'note_shared_status', 'name', 'callee', 'Mode', 'recipient', 'list_items', 'Number', 'app_developer', 'message', 'event_performer', 'list_app', 'delay', 'note_property', 'component', 'GameLevel', 'parameter', 'note_app', 'distance', 'content', 'payment_communication_mode', 'commercial_provider', 'delivery_time', 'meal', 'subject', 'organization', 'family_member', 'duration', 'list_label', 'trigger_time_status', 'business_type', 'GenericDescription', 'list_quantifier', 'menu_type', 'list_title', 'list_singular_or_plural', 'pickup_location', 'resolution', 'location', 'number', 'datetime', 'security_identifier', 'food', 'event_location', 'document_format', 'price_type', 'business_hours', 'activity', 'order_placement_date', 'order_item', 'Device', 'restaurant_location', 'order_number', 'device', 'parking_type', 'trigger_time', 'business_type_label', 'usualness', 'note_quantifier', 'number_of_repetitions', 'list_owner', 'product', 'dropoff_time', 'event_performance', 'contact_id', 'list_number', 'service_type', 'frame_rate', 'payment_method', 'trading_session', 'topic', 'unit', 'dropoff_location', 'pickup_time', 'health_stat_type', 'event_time', 'person', 'title', 'filter', 'card_company', 'emergency_contact_type', 'note_assignee', 'list_position', 'culinary_quantity', 'business_prices', 'event_type', 'format', 'parking_structure', 'Provider', 'absolute_position', 'time', 'note_content', 'medium', 'provider', 'items', 'app_phrase', 'relative_position', 'contact_id_type', 'time_span', 'list_creation_time', 'billing_date_span', 'attachment', 'note_number', 'instrument', 'GameName', 'id_form', 'phone_number', 'note_creation_time', 'Operand', 'note_singular_or_plural', 'note_label', 'start_time', 'app', 'menu_item', 'amount', 'health_stat_domain', 'note_shared_group', 'modality', 'restaurant', 'number_of_sets', 'account_provider', 'billing_cycle', 'billing_frequency', 'label', 'ticket_quantity', 'Opponent', 'size', 'delivery_method', 'department', 'payment_amount', 'end_time', 'product_source', 'cuisine_type', 'security_type', 'stock_market', 'polarity', 'date', 'app_category']

input_file = open(sys.argv[1], 'r')
output_file = open(sys.argv[2], 'w')

lines = input_file.readlines()

for index, line in enumerate(lines):
    if index % 100 == 0:
        print(index)
    word_list = line.split()
    new_sentence = []
    skip_next = False
    for i in range(len(word_list)):
        if skip_next:
            skip_next = False
            continue
        curr_word = word_list[i]
        new_word = word_list[i]
        if i != len(word_list) - 1 and (curr_word[-1] == '_' or curr_word[-1] == '_'):
            curr_word += word_list[i + 1]
            new_word += word_list[i + 1]
            skip_next = True
        elif i != len(word_list) - 1 and (curr_word == 'Infer' and word_list[i + 1] == 'fromContext'):
            curr_word = 'InferFromContext'
            new_word = 'InferFromContext'
            skip_next = True
        elif 'Infer' in curr_word:
            curr_word = 'InferFromContext'
            new_word = 'InferFromContext'
        if i != len(word_list) - 1 and word_list[i + 1] == '(':
            if curr_word not in intents:
                min_dist = 1e9
                nearest_word = ''
                for intent in intents:
                    dist = edit_distance(curr_word, intent)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_word = intent
                new_word = nearest_word
                print(f'Index {index}: Original Word {curr_word} changed to {new_word}')
        elif i != len(word_list) - 1 and word_list[i + 1] == '«':
            if curr_word not in slots:
                if curr_word == 'people':
                    new_word = 'person'
                else:
                    min_dist = 1e9
                    nearest_word = ''
                    for slot in slots:
                        dist = edit_distance(curr_word, slot)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_word = slot
                    new_word = nearest_word
                print(f'Index {index}: Original Word {curr_word} changed to {new_word}')
        new_sentence.append(new_word)
    new_sentence = ' '.join(new_sentence)
    output_file.write(new_sentence + '\n')

input_file.close()
output_file.close()