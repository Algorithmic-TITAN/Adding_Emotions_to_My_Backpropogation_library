class Gradient_decent_algorithm_with_add_ons:
    global BASE_LEARNING_RATE
    global activation_function_prescision
    global past_node_errors
    global past_network_errors
    global past_networks
    global LEARNING_RATE
    global MAX_LEARNING_RATE
    MAX_LEARNING_RATE=0.001
    LEARNING_RATE=0.00001
    past_network_errors=[]
    past_networks=[]
    BASE_LEARNING_RATE=0.001
    activation_function_prescision=10
    past_node_errors=[]

    def init(layers, initializing_range):
      import random
      global weights_amount
      global full_population_weights
      global full_population_biases
      global weights_in_the_layers_so_far
      global nodes_counted_in_each_layer
      global nodes_amount
      nodes_counted_in_each_layer=[0]
      weights_in_the_layers_so_far=[]
      weights_amount=0
      nodes_amount=layers[0]
      nodes_counted_in_each_layer.append(layers[0])
      weights_in_the_layers_so_far.append(0)
      for i1 in range(len(layers)-1):
        weights_amount+=layers[i1]*layers[i1+1]
        nodes_amount+=layers[i1+1]
        weights_in_the_layers_so_far.append(weights_amount)
        nodes_counted_in_each_layer.append(nodes_amount)
      full_population_weights=[]
      full_population_biases=[]
      for i3 in range(weights_amount):
        full_population_weights.append(round(random.uniform(initializing_range[0], initializing_range[1]), 3))
      nodes_amount=nodes_amount-layers[0]
      for i4 in range(nodes_amount):
        full_population_biases.append(round(random.uniform(initializing_range[0], initializing_range[1]), 3))

    def run_network(INPUTS, layers, activation_functions_being_used):
        global outputs
        global node_firing_numbers
        global weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer
        activation_functions_being_used=list(map(str.lower, activation_functions_being_used))
        if "sigmoid" in activation_functions_being_used:
          sigmoid_being_used_in_run=True
        else:
          sigmoid_being_used_in_run=False
        if "binarystep" in activation_functions_being_used or "binary_step" in activation_functions_being_used:
            binary_step_being_used_in_run=True
        else:
            binary_step_being_used_in_run=False
        running_network_weights=full_population_weights
        running_network_biases=full_population_biases
        node_firing_numbers=[]
        weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer=[]
        NODES_PROSCESSED=INPUTS
        for i in range(len(INPUTS)):
          node_firing_numbers.append(INPUTS[i])
        nodes_proscessing=NODES_PROSCESSED
        computatinoal_part=0
        computatinoal_part_biases=0
        for i1 in range(len(layers)-1):
            nodes_proscessing=[]
            weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer.append([])
            for i2 in range(layers[i1+1]):
                node_in_next_layer_VALUE=0
                weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer[i1].append([])
                for i3 in range(layers[i1]):
                    edge_value=running_network_weights[computatinoal_part]*NODES_PROSCESSED[i3]
                    weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer[i1][i2].append(running_network_weights[computatinoal_part])
                    computatinoal_part+=1
                    node_in_next_layer_VALUE+=edge_value
                nodes_proscessing.append(node_in_next_layer_VALUE+running_network_biases[computatinoal_part_biases])
                node_firing_numbers.append(node_in_next_layer_VALUE+running_network_biases[computatinoal_part_biases])
                computatinoal_part_biases+=1
            NODES_PROSCESSED=nodes_proscessing
        outputs=NODES_PROSCESSED
        import decimal
        import math
        decimal.getcontext().prec=activation_function_prescision+1
        if sigmoid_being_used_in_run:
            for i4 in range(len(outputs)):
                outputs[i4]=float(round(decimal.Decimal(decimal.Decimal(1)/(decimal.Decimal(1)+decimal.Decimal(2.71828)**(decimal.Decimal(-1)*(decimal.Decimal(outputs[i4]))))),activation_function_prescision))
        elif binary_step_being_used_in_run:
            for i5 in range(len(outputs)):
                if outputs[i5]>=0:
                    outputs[i5]=1
                else:
                    outputs[i5]=0
        if "tanh" in activation_functions_being_used or "hyperbolic_tangent" in activation_functions_being_used or "hyperbolictangent" in activation_functions_being_used:
            for i6 in range(len(outputs)):
                outputs[i6]=float(round(decimal.Decimal((((decimal.Decimal(decimal.Decimal(2.71828)**decimal.Decimal(outputs[i6])))-(decimal.Decimal(decimal.Decimal(2.71828)**(decimal.Decimal(decimal.Decimal(-1)*decimal.Decimal(outputs[i6]))))))))/decimal.Decimal((((decimal.Decimal(decimal.Decimal(2.71828)**decimal.Decimal(outputs[i6])))+(decimal.Decimal(decimal.Decimal(2.71828)**(decimal.Decimal(decimal.Decimal(-1)*decimal.Decimal(outputs[i6])))))))), activation_function_prescision))
        if "softplus" in activation_functions_being_used:
            for i7 in range(len(outputs)):
                if outputs[i7]>2608:
                    outputs[i7]=2607
                if outputs[i7]<-2608:
                    outputs[i7]=-2607
            def add_precicion_softplus(output_number_softplus):
                decimal.getcontext().prec+=1
                try:
                    outputs[i8]=float(round(math.log(1+decimal.Decimal(2.71828)**decimal.Decimal(outputs[i8])),activation_function_prescision))
                except:
                    add_precicion_softplus(output_number_softplus)
            for i8 in range(len(outputs)):
                try:
                 outputs[i8]=float(round(math.log(1+decimal.Decimal(2.71828)**decimal.Decimal(outputs[i8])),activation_function_prescision))
                except:
                    add_precicion_softplus(i8)
                    outputs[i8]=float(round(math.log(1+decimal.Decimal(2.71828)**decimal.Decimal(outputs[i8])),activation_function_prescision))
        if "gaussian" in activation_functions_being_used or "gaussian_function" in activation_functions_being_used or "gaussianfunction" in activation_functions_being_used:

            for i9 in range(len(outputs)):
                outputs[i9]=2.71828**-outputs[i9]**2
        return(outputs)

    def split_data_TRAINING_TESTING(RATIO, DATA):
      global TRAINING_DATA
      global TESTING_DATA
      TRAINING_DATA=[]
      TESTING_DATA=[]
      if RATIO[0]+RATIO[1]==1:
        for i in range(int(round((len(DATA)/(1/RATIO[0])), 0))):
          TESTING_DATA.append(DATA[i+len(DATA)-(int(round((len(DATA)/(1/RATIO[0])), 0)))])
        for i1 in range(len(DATA)-(int(round((len(DATA)/(1/RATIO[0])), 0)))):
          TRAINING_DATA.append(DATA[i1])
      else:
        raise Exception('This is not a valid percentage ratio. Please enter a ratio that adds up to 1.')

    def find_cost(output, goals):
      global costs
      costs=[]
      import decimal
      decimal.getcontext().prec=1000
      for i in range(len(output)):
        costs.append(float((decimal.Decimal(output[i])-decimal.Decimal(goals[i]))**decimal.Decimal(2)))
      return costs

    def test(test_data, layers, activation_functions):
      global avg_error
      error=0 
      for i in range(len(test_data)):
        for i1 in range(len(test_data[i][1])):
          Gradient_decent_algorithm_with_add_ons.run_network(test_data[i][0], layers, activation_functions)
          goal=test_data[i][1][i1]
          output=outputs
          error+=((goal-output[i1])**2)
      avg_error=error/(len(test_data[i][1])*len(test_data))
      return avg_error




    def take_gradient_decent_step(layers, activation_functions, TRAINING_DATA):
      global full_population_biases
      global full_population_weights
      activation_functions_filtered=list(map(str.lower, activation_functions))

      def decide_what_way_to_nudge_values(layers, DATA_INSTANCE, activation_functions_being_used_weights_improvement):
          global derivatives_in_network_weights
          global all_derivatives_in_network_biases
          import decimal
          try:
            activation_functions_being_used_weights_improvement.pop(activation_functions_being_used_weights_improvement.index('binarystep'))
            Gradient_decent_algorithm_with_add_ons.run_network(TRAINING_DATA[DATA_INSTANCE][0],layers, activation_functions_being_used_weights_improvement)
            activation_functions_being_used_weights_improvement.append('binarystep')
          except:
              try:
                activation_functions_being_used_weights_improvement.pop(activation_functions_being_used_weights_improvement.index('binary_step'))
                Gradient_decent_algorithm_with_add_ons.run_network(TRAINING_DATA[DATA_INSTANCE][0],layers, activation_functions_being_used_weights_improvement)
                activation_functions_being_used_weights_improvement.append('binary_step')
              except:
                  activation_functions_being_used_weights_improvement_FILTERED_FOR_BINARY_STEP=activation_functions_being_used_weights_improvement
                  Gradient_decent_algorithm_with_add_ons.run_network(TRAINING_DATA[DATA_INSTANCE][0],layers, activation_functions_being_used_weights_improvement)
          last_layer_to_cost_effects=[]
          weight_surrounding_layer_numbers=[]          
          for i3 in range(layers[len(layers)-1]):
            decimal.getcontext().prec=activation_function_prescision+1
            current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=round(float(decimal.Decimal(2)*(decimal.Decimal(outputs[i3])-decimal.Decimal(TRAINING_DATA[DATA_INSTANCE][1][i3]))), activation_function_prescision)
            if 'sigmoid' in activation_functions_being_used_weights_improvement:
                current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=round(current_node_derivative_appending_to_originoal_last_layer_to_cost_effects*round(float((decimal.Decimal(1)/(decimal.Decimal(1)+decimal.Decimal(2.71828)**(decimal.Decimal(-1)*(decimal.Decimal(outputs[i3])))))*(decimal.Decimal(1)-((decimal.Decimal(1)/(decimal.Decimal(1)+decimal.Decimal(2.71828)**(decimal.Decimal(-1)*(decimal.Decimal(outputs[i3])))))))), activation_function_prescision), activation_function_prescision)
            if 'binary_step' in activation_functions_being_used_weights_improvement or 'binarystep' in activation_functions_being_used_weights_improvement:
                current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=round(float(decimal.Decimal(2)*(decimal.Decimal(outputs[i3])-(decimal.Decimal(TRAINING_DATA[DATA_INSTANCE][1][i3])*200-100))), activation_function_prescision)
            if "tanh" in activation_functions_being_used_weights_improvement or "hyperbolic_tangent" in activation_functions_being_used_weights_improvement or "hyperbolictangent" in activation_functions_being_used_weights_improvement:
                current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=round(current_node_derivative_appending_to_originoal_last_layer_to_cost_effects*float(round(1-decimal.Decimal(float(round(decimal.Decimal((((decimal.Decimal(decimal.Decimal(2.71828)**decimal.Decimal(outputs[i3])))-(decimal.Decimal(decimal.Decimal(2.71828)**(decimal.Decimal(decimal.Decimal(-1)*decimal.Decimal(outputs[i3]))))))))/decimal.Decimal((((decimal.Decimal(decimal.Decimal(2.71828)**decimal.Decimal(outputs[i3])))+(decimal.Decimal(decimal.Decimal(2.71828)**(decimal.Decimal(decimal.Decimal(-1)*decimal.Decimal(outputs[i3])))))))), activation_function_prescision)))**2, activation_function_prescision)), activation_function_prescision)
            if "softplus" in activation_functions_being_used_weights_improvement:
                current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=round(current_node_derivative_appending_to_originoal_last_layer_to_cost_effects*float(decimal.Decimal(1)/(decimal.Decimal(1)+decimal.Decimal(2.71828)**(decimal.Decimal(-1)*(decimal.Decimal(outputs[i3]))))), activation_function_prescision)
            if "gaussian" in activation_functions_being_used_weights_improvement or "gaussian_function" in activation_functions_being_used_weights_improvement or "gaussianfunction" in activation_functions_being_used_weights_improvement:
                import math
                current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=current_node_derivative_appending_to_originoal_last_layer_to_cost_effects*float(decimal.Decimal(-2)*decimal.Decimal(outputs[i3])*((decimal.Decimal(math.e)**decimal.Decimal(-outputs[i3]))**decimal.Decimal(2)))
            last_layer_to_cost_effects.append(current_node_derivative_appending_to_originoal_last_layer_to_cost_effects)
            last_layer_to_cost_effects_LEN=len(last_layer_to_cost_effects)
          biases_went_back_counter=0
          derivatives_in_network_weights=[]
          all_derivatives_in_network_biases=[]
          for i2 in range(len(LAYERS_BEING_USED)-1):
            for i4 in range(layers[len(layers)-i2-1]):
              if i2==0:
                  influence_amount="NaN"
              else:
                  influence_amount=layers[len(layers)-i2]
              summed_influence_on_next_layer=0

              if not influence_amount=="NaN":
               for i5 in range(influence_amount):
                weight_carrying_effect_value=weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer[len(layers)-i2-1][i5][i4]
                effects_in_the_last_layer=weight_carrying_effect_value*last_layer_to_cost_effects[i5]
                summed_influence_on_next_layer+=effects_in_the_last_layer
               last_layer_to_cost_effects.append(summed_influence_on_next_layer)
              else:
               for i5 in range(1):
                  effect_of_bias=last_layer_to_cost_effects[i4]
               last_layer_to_cost_effects.append(effect_of_bias)
               biases_went_back_counter+=1
            for i7 in range(last_layer_to_cost_effects_LEN):
                last_layer_to_cost_effects.pop(0)
            for i8 in range(len(last_layer_to_cost_effects)):
              for i9 in range(layers[len(layers)-i2-2]):
                derivatives_in_network_weights.append(last_layer_to_cost_effects[(len(last_layer_to_cost_effects)-i8-1)]*node_firing_numbers[(layers[len(layers)-i2-2]-i9-1)+nodes_counted_in_each_layer[len(layers)-i2-2]])
            for i10 in range(len(last_layer_to_cost_effects)):
                all_derivatives_in_network_biases.append(last_layer_to_cost_effects[i8])
            last_layer_to_cost_effects_LEN=len(last_layer_to_cost_effects)
          all_derivatives_in_network_biases.reverse()
          derivatives_in_network_weights.reverse()

      
      #This is all just commented out in honor of times when this code was in production and in use

      #def decide_what_way_to_nudge_biases(DATA_INSTANCE, activation_functions_being_used_bias_improvement):
      #    import decimal
      #    try:
      #      activation_functions_being_used_bias_improvement.pop(activation_functions_being_used_bias_improvement.index('binarystep'))
      #      Gradient_decent_algorithm_with_add_ons.run_network(TRAINING_DATA[DATA_INSTANCE][0],layers, activation_functions_being_used_bias_improvement)
      #      activation_functions_being_used_bias_improvement.append('binarystep')
      #    except:
      #        try:
      #          activation_functions_being_used_bias_improvement.pop(activation_functions_being_used_bias_improvement.index('binary_step'))
      #          Gradient_decent_algorithm_with_add_ons.run_network(TRAINING_DATA[DATA_INSTANCE][0],layers, activation_functions_being_used_bias_improvement)
      #          activation_functions_being_used_bias_improvement.append('binary_step')
      #        except:
      #            activation_functions_being_used_bias_improvement_FILTERED_FOR_BINARY_STEP=activation_functions_being_used_bias_improvement
      #            Gradient_decent_algorithm_with_add_ons.run_network(TRAINING_DATA[DATA_INSTANCE][0],layers, activation_functions_being_used_bias_improvement)
      #    last_layer_to_cost_effects=[]
      #    for i3 in range(layers[len(layers)-1]):
      #      decimal.getcontext().prec=activation_function_prescision+1
      #      current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=round(float(decimal.Decimal(2)*(decimal.Decimal(outputs[i3])-decimal.Decimal(TRAINING_DATA[DATA_INSTANCE][1][i3]))), activation_function_prescision)
      #     if 'sigmoid' in activation_functions_being_used_bias_improvement:
      #          current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=round(current_node_derivative_appending_to_originoal_last_layer_to_cost_effects*round(float((decimal.Decimal(1)/(decimal.Decimal(1)+decimal.Decimal(2.71828)**(decimal.Decimal(-1)*(decimal.Decimal(outputs[i3])))))*(decimal.Decimal(1)-((decimal.Decimal(1)/(decimal.Decimal(1)+decimal.Decimal(2.71828)**(decimal.Decimal(-1)*(decimal.Decimal(outputs[i3])))))))), activation_function_prescision), activation_function_prescision)
      #      if 'binary_step' in activation_functions_being_used_bias_improvement or 'binarystep' in activation_functions_being_used_bias_improvement:
      #              current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=round(float(decimal.Decimal(2)*(decimal.Decimal(outputs[i3])-(decimal.Decimal(TRAINING_DATA[DATA_INSTANCE][1][i3])*200-100))), activation_function_prescision)
      #      if "tanh" in activation_functions_being_used_bias_improvement or "hyperbolic_tangent" in activation_functions_being_used_bias_improvement or "hyperbolictangent" in activation_functions_being_used_bias_improvement:
      #          current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=round(current_node_derivative_appending_to_originoal_last_layer_to_cost_effects*float(round(1-decimal.Decimal(float(round(decimal.Decimal((((decimal.Decimal(decimal.Decimal(2.71828)**decimal.Decimal(outputs[i3])))-(decimal.Decimal(decimal.Decimal(2.71828)**(decimal.Decimal(decimal.Decimal(-1)*decimal.Decimal(outputs[i3]))))))))/decimal.Decimal((((decimal.Decimal(decimal.Decimal(2.71828)**decimal.Decimal(outputs[i3])))+(decimal.Decimal(decimal.Decimal(2.71828)**(decimal.Decimal(decimal.Decimal(-1)*decimal.Decimal(outputs[i3])))))))), activation_function_prescision)))**2, activation_function_prescision)), activation_function_prescision)
      #     if "softplus" in activation_functions_being_used_bias_improvement:
      #          current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=round(current_node_derivative_appending_to_originoal_last_layer_to_cost_effects*float(decimal.Decimal(1)/(decimal.Decimal(1)+decimal.Decimal(2.71828)**(decimal.Decimal(-1)*(decimal.Decimal(outputs[i3]))))), activation_function_prescision)
      #      if "gaussian" in activation_functions_being_used_bias_improvement or "gaussian_function" in activation_functions_being_used_bias_improvement or "gaussianfunction" in activation_functions_being_used_bias_improvement:
      #          import math
      #          current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=current_node_derivative_appending_to_originoal_last_layer_to_cost_effects*float(decimal.Decimal(-2)*decimal.Decimal(outputs[i3])*((decimal.Decimal(math.e)**decimal.Decimal(-outputs[i3]))**decimal.Decimal(2)))
      #      last_layer_to_cost_effects.append(current_node_derivative_appending_to_originoal_last_layer_to_cost_effects)
      #    last_layer_to_cost_effects_LEN=len(last_layer_to_cost_effects)
      #    biases_went_back_counter=0
      #    for i2 in range(len(layers)-1):
      #      for i4 in range(layers[len(layers)-i2-1]):
      #        if i2==0:
      #            influence_amount="NaN"
      #        else:
      #            influence_amount=layers[len(layers)-i2]
      #        summed_influence_on_next_layer=0
      #
      #        if not influence_amount=="NaN":
      #         for i5 in range(influence_amount):
      #          weight_carrying_effect_value=weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer[len(layers)-i2-1][i5][i4]
      #          effects_in_the_last_layer=weight_carrying_effect_value*last_layer_to_cost_effects[i5]
      #          summed_influence_on_next_layer+=effects_in_the_last_layer
      #         last_layer_to_cost_effects.append(summed_influence_on_next_layer)
      #        else:
      #         for i5 in range(1):
      #            effect_of_bias=last_layer_to_cost_effects[i4]
      #         last_layer_to_cost_effects.append(effect_of_bias)
      #         biases_went_back_counter+=1
      #
      #      if i2==0:
      #         for i6 in range(layers[len(layers)-1]):
      #          last_layer_to_cost_effects.pop(0)
      #         last_layer_to_cost_effects_LEN=len(last_layer_to_cost_effects)
      #      else:
      #         for i7 in range(layers[len(layers)-i2]):
      #          last_layer_to_cost_effects.pop(0)
      #         last_layer_to_cost_effects_LEN=len(last_layer_to_cost_effects)
      

      for i in range(len(TRAINING_DATA)):
        decide_what_way_to_nudge_values(LAYERS_BEING_USED, i, activation_functions_filtered)
        for i1 in range(len(derivatives_in_network_weights)):
            full_population_weights[i1]+=derivatives_in_network_weights[i1]*LEARNING_RATE*-1/len(TRAINING_DATA)
        for i2 in range(len(all_derivatives_in_network_biases)):
            full_population_biases[i2]+=all_derivatives_in_network_biases[i2]*LEARNING_RATE*-1/len(TRAINING_DATA)



    def set_activation_function_prescision(prescicion):
        global activation_function_prescision
        activation_function_prescision=prescicion


    def do_all_with_determination(data, layers, activation_functions, amount_of_determination, determination_prescicion):
        import random
        question_error_scores=[]
        for i in range(len(TRAINING_DATA)):
            question_error_scores.append(Gradient_decent_algorithm_with_add_ons.find_cost(Gradient_decent_algorithm_with_add_ons.run_network(TRAINING_DATA[i][0], layers, activation_functions), TRAINING_DATA[i][1]))
        full_question_problematic_scores=[]
        for i1 in range(len(TRAINING_DATA)):
            full_question_problematic_scores.append(sum(question_error_scores[i1]))
        sum_of_all_problematic_things=sum(full_question_problematic_scores)
        probabilities_list=[]
        for i3 in range(len(full_question_problematic_scores)):
            probabilities_list.append(round(full_question_problematic_scores[i3]/sum_of_all_problematic_things+0.1,determination_prescicion))
        probabalistic_list=[]
        for i4 in range(len(probabilities_list)):
            for i5 in range(int(10**determination_prescicion*probabilities_list[i4])):
                probabalistic_list.append(i4)
        for i6 in range(amount_of_determination):
            random_choice_to_focus_on=random.randint(0,10**determination_prescicion-1)
            Gradient_decent_algorithm_with_add_ons.take_gradient_decent_step(layers, activation_functions, TRAINING_DATA)
            


    def use_anger_ability(layers, activation_functions, lowest_improvement_per_check_acceptance, print_all, coolheadedness):
        global past_node_errors
        global past_network_errors
        global past_networks
        for i in range(len(TRAINING_DATA)):
            past_node_errors.append(sum(Gradient_decent_algorithm_with_add_ons.find_cost(Gradient_decent_algorithm_with_add_ons.run_network(TRAINING_DATA[i][0], layers, activation_functions), TRAINING_DATA[i][1]))/layers[len(layers)-1])
            if len(past_node_errors)>len(TRAINING_DATA):
              past_node_errors.pop(0)
        past_network_errors.append(sum(past_node_errors)/len(TRAINING_DATA))
        if len(past_network_errors)>coolheadedness:
            past_network_errors.pop(0)
        times_with_invalid_change=0
        for i2 in range(len(past_network_errors)-1):
          try:
             if past_network_errors[len(past_network_errors)-1-i2]>=past_network_errors[len(past_network_errors)-2-i2]-lowest_improvement_per_check_acceptance:
                    times_with_invalid_change+=1
          except:
              pass
        if times_with_invalid_change==coolheadedness-1:
            past_networks.append([full_population_weights, full_population_biases])
            past_network_errors.append(Gradient_decent_algorithm_with_add_ons.test(TRAINING_DATA, layers, []))
            Gradient_decent_algorithm_with_add_ons.init(LAYERS_BEING_USED,  [-3,3])
            past_network_errors=[]
        if print_all:
            try:
                print('The best weights so far are: ', past_networks[past_network_errors.index(min(past_network_errors))][0])
                print('The best biases so far are: ', past_networks[past_network_errors.index(min(past_network_errors))][1])
            except:
                print('The best weights so far are: ', full_population_weights)
                print('The best biases so far are: ', full_population_biases)


    def use_excitement_ability(layers, activation_functions_being_used):
        global LEARNING_RATE
        error_scores=[]
        for i in range(len(TRAINING_DATA)):
            error_scores.append(sum(Gradient_decent_algorithm_with_add_ons.find_cost(Gradient_decent_algorithm_with_add_ons.run_network(TRAINING_DATA[i][0], layers, activation_functions_being_used), TRAINING_DATA[i][1]))/layers[len(layers)-1])
        LEARNING_RATE=(2.71828**-((sum(error_scores)/len(TRAINING_DATA)))**2+BASE_LEARNING_RATE)*MAX_LEARNING_RATE


        
    def auto_for_with_determination(layers, iterations, activation_functions, data, amount_of_determination, determination_prescicion):
        for i in range(iterations): #This loop is the loop that makes it go through multiple epochs
            #Gradient_decent_algorithm_with_add_ons.take_gradient_decent_step(layers, activation_functions, TRAINING_DATA) #this does all things improvement-related
            Gradient_decent_algorithm_with_add_ons.do_all_with_determination(data, layers, activation_functions, amount_of_determination, determination_prescicion)


    def auto_for_with_emotions(layers, iterations, activation_functions, data, amount_of_emotion, emotion_prescicion, lowest_improvement_per_check_acceptance, print_all, coolheadedness):
      for i in range(iterations):
        #anger, which controls getting out of the local minimum and prints the best weights and biases so far
        Gradient_decent_algorithm_with_add_ons.use_anger_ability(layers, activation_functions, lowest_improvement_per_check_acceptance, print_all, coolheadedness)
        #excitement, which adjusts the learning rate to give more precsice results faster
        Gradient_decent_algorithm_with_add_ons.use_excitement_ability(layers, activation_functions)
           

data=[[[3],[2,3,4,5,6]],[[1],[0,1,2,3,4]],[[2],[1,2,3,4,5]]]
LAYERS_BEING_USED=[1,5]
Gradient_decent_algorithm_with_add_ons.init(LAYERS_BEING_USED,  [-3,3])
Gradient_decent_algorithm_with_add_ons.split_data_TRAINING_TESTING([0.1,0.9], data)
Gradient_decent_algorithm_with_add_ons.set_activation_function_prescision(1000)
for i in range(1000000000):
    Gradient_decent_algorithm_with_add_ons.auto_for_with_emotions(LAYERS_BEING_USED, 1, [''], data, 1, 2, 0.000000001, True, 10000) #this is computatinoally expensive, so I highly advise you don't use it every iteration
    Gradient_decent_algorithm_with_add_ons.auto_for_with_determination(LAYERS_BEING_USED, 1, [''], data, 1000, 2) #this function is computatinoally expensive at the start, so I highly reccomend that you put the amount of iterations high. 
    print(Gradient_decent_algorithm_with_add_ons.run_network(TRAINING_DATA[0][0], LAYERS_BEING_USED, [])) #GET RID OF THIS WHEN ALL TESTING DONE