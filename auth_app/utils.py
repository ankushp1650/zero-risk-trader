# from collections import defaultdict
# from auth_app.models import Transaction, UserTrade
# from decimal import Decimal
#
#
# def recommend_stocks_total_based(user_trades_queryset):
#     if not user_trades_queryset:
#         return []  # Or return [{'stock': 'N/A', 'profit_percent': 0, 'success_rate': 0}]
#
#     profit_by_stock = defaultdict(list)
#
#     for trade in user_trades_queryset:
#         investment_value = trade.buy_price * trade.quantity
#         current_value = trade.sell_price * trade.quantity
#         profit_percentage = ((current_value - investment_value) / investment_value) * 100
#         profit_by_stock[trade.stock_symbol].append(profit_percentage)
#
#     avg_profit = {}
#     success_rate = {}
#
#     for stock, profits in profit_by_stock.items():
#         avg_profit[stock] = sum(profits) / len(profits)
#         profitable_trades = [p for p in profits if p > 0]
#         success_rate[stock] = (len(profitable_trades) / len(profits)) * 100
#
#     sorted_stocks = sorted(avg_profit.items(), key=lambda x: x[1], reverse=True)
#
#     recommendations = []
#     for stock, _ in sorted_stocks[:2]:
#         recommendations.append({
#             'stock': stock,
#             'profit_percent': round(avg_profit[stock], 2),
#             'success_rate': round(success_rate[stock], 2),
#         })
#
#     print("Recommendations:", recommendations)  # Debugging: Check the recommendations list
#
#     return recommendations
#
#
# def transfer_transactions_to_user_trade():
#     transactions = Transaction.objects.all().order_by('date')
#
#     user_buy_transactions = {}
#
#     for transaction in transactions:
#         user = transaction.user
#         stock = transaction.stock
#         quantity = transaction.quantity
#         price_at_transaction = transaction.price_at_transaction
#         transaction_type = transaction.transaction_type
#         transaction_date = transaction.date
#         transaction_cost = transaction.cost or Decimal('0.00')  # Handle if cost is None
#
#         if transaction_type == 'BUY':
#             if user.id not in user_buy_transactions:
#                 user_buy_transactions[user.id] = {}
#
#             if stock not in user_buy_transactions[user.id]:
#                 user_buy_transactions[user.id][stock] = []
#
#             user_buy_transactions[user.id][stock].append({
#                 'quantity': quantity,
#                 'price': price_at_transaction,
#                 'date': transaction_date,
#                 'cost': transaction_cost
#             })
#
#         elif transaction_type == 'SELL':
#             if user.id in user_buy_transactions and stock in user_buy_transactions[user.id]:
#                 buy_transactions = user_buy_transactions[user.id][stock]
#
#                 while buy_transactions and quantity > 0:
#                     buy_transaction = buy_transactions.pop(0)
#                     buy_quantity = buy_transaction['quantity']
#                     buy_price = buy_transaction['price']
#                     buy_date = buy_transaction['date']
#                     buy_cost = buy_transaction['cost']
#
#                     if quantity < buy_quantity:
#                         # Partial match
#                         total_cost = buy_cost + transaction_cost  # Sum of buy and sell costs
#                         UserTrade.objects.create(
#                             user=user,
#                             stock_symbol=stock,
#                             buy_price=buy_price,
#                             sell_price=price_at_transaction,
#                             quantity=quantity,
#                             buy_date=buy_date,
#                             sell_date=transaction_date,
#                             cost=total_cost
#                         )
#                         remaining_quantity = buy_quantity - quantity
#                         if remaining_quantity > 0:
#                             buy_transactions.insert(0, {
#                                 'quantity': remaining_quantity,
#                                 'price': buy_price,
#                                 'date': buy_date,
#                                 'cost': buy_cost  # Cost remains same for remaining
#                             })
#                         break
#                     else:
#                         # Full match
#                         total_cost = buy_cost + transaction_cost  # Sum of buy and sell costs
#                         UserTrade.objects.create(
#                             user=user,
#                             stock_symbol=stock,
#                             buy_price=buy_price,
#                             sell_price=price_at_transaction,
#                             quantity=buy_quantity,
#                             buy_date=buy_date,
#                             sell_date=transaction_date,
#                             cost=total_cost
#                         )
#                         quantity -= buy_quantity
#
#                 if quantity > 0:
#                     print(f"Warning: Not enough BUY transactions to cover SELL of {quantity} stocks.")
#
#
# # utils/date_parser.py (create this file if needed)
#
# from datetime import datetime
#
#
# def parse_iso_datetime(date_str):
#     """
#     Safely parse an ISO-formatted datetime string.
#     Returns a `datetime` object or None if parsing fails.
#     """
#     if not date_str or not isinstance(date_str, str):
#         return None
#
#     try:
#         return datetime.fromisoformat(date_str)
#     except ValueError:
#         # Handles formats like '2024-05-04 15:32:00' as well
#         try:
#             return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
#         except ValueError:
#             return None
