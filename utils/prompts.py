plot_prompt = """ Extract the data that is needed to create a plot as requested below. 
                
                
                "{}"

                Present your respone as CSV string. Add <CSV> at the begining and end of the CSV string
                example of reponse
                <CSV>
                col1,col2
                a,b
                c,d
                <CSV>
        """