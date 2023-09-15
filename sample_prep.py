


def get_procedure(concentrations, mL_stock, mL_dilutions):
    biggest_conc = max(concentrations)

    # GSH
    g_GSH = biggest_conc * .001 * .001 * 307.32 * mL_stock

    # GSSG
    g_GSSG = biggest_conc * .001 * .001 * 614.64 * mL_stock

    print('dissolve', g_GSH, 'g GSH in', mL_stock, 'mL buffer to make the stock solution')
    print('dissolve', g_GSSG, 'g GSSG in', mL_stock, 'mL buffer to make the stock solution')

    # dilutions
    stocks = []
    for i in concentrations:
        ratio = i/biggest_conc
        ratio = ratio*mL_dilutions
        micro_liters = ratio*1000
        mL_dilutions1 = mL_dilutions*1000
        print(i, 'mM:', micro_liters, 'um stock in', mL_dilutions1-micro_liters, 'um buffer')
        stocks.append(micro_liters)

    print(stocks)


    return None



if __name__ == '__main__':
    print(get_procedure([50,25,33,32,31], 2, .2))

