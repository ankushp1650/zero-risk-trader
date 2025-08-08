from datetime import datetime, date
from decimal import Decimal
import pandas as pd
from django.utils.timezone import now
from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.db.models import Max, Sum, OuterRef
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout
from auth_project.settings import EMAIL_HOST_USER
from .aiml.prediction_models import train_linear_model, predict_next_day, train_decision_tree_model, \
    train_random_forest_model, train_svm_model, predict_next_day_svm, train_lstm_model, predict_next_day_lstm, \
    save_lstm_to_db, load_lstm_from_db, calculate_model_scores, calculate_success_rate, \
    calculate_directional_success_rate, calculate_avg_error
from .custom_utils.explainability import generate_model_explainability
from .custom_utils.fetching import get_daily_time_series
from .custom_utils.graphs import generate_graphs, bar_chart_view, pie_chart_view, line_chart_view, quantity_bar_graph, \
    pnl_bar_chart_view
from .forms import RegisterForm
from .models import Portfolio, Transaction, UserProfile, Final_holding, StockJason, CurrentPrice, StockPrediction, \
    StockPerformance, BestModelRecord \
    # , StockPrediction, \
# BestStockRecommendation
from django.contrib.auth import login
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.urls import reverse, reverse_lazy
from django.contrib.auth.views import PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, \
    PasswordResetCompleteView
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail
from django.conf import settings
from django.utils.http import urlsafe_base64_decode
from django.utils.encoding import force_str
from django.contrib.auth.tokens import default_token_generator
from django.contrib import messages
from django.shortcuts import redirect
from django.contrib.auth.models import User
from django.http import HttpResponseBadRequest, HttpResponse
# from django.shortcuts import render
# from .models import UserTrade
# from .utils import recommend_stocks_total_based, transfer_transactions_to_user_trade, parse_iso_datetime

import logging
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

User = get_user_model()  # Get the user model in case you're using a custom user model


# @guest
# def login_view(request):
#     if request.method == 'POST':
#         form = AuthenticationForm(request, data=request.POST)
#         if form.is_valid():
#             user = form.get_user()
#             login(request, user)
#             # return redirect('api_key_form')
#             return redirect('save_api_key')
#     else:
#         form = AuthenticationForm()
#     return render(request, 'auth/login.html', {'form': form})


def login_view(request):
    print("Login view triggered!")
    if request.user.is_authenticated:
        print("👤 User already authenticated, redirecting to dashboard.")
        return redirect('dashboard')

    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            print(f"✅ User '{user.username}' logged in Successfully ")

            login(request, user)

            # 👇 Check if API key exists in UserProfile
            try:
                user_profile, created = UserProfile.objects.get_or_create(user=user)

                # ✨ Check if today’s data has already been fetched
                today = now().date()
                if user_profile.last_data_fetch_date != today:
                    # 🛠️ Fetch fresh data
                    print("📥 Fetching new data for user...")
                    fetch_and_save_user_data(request)
                    # Update last_data_fetch_date
                    user_profile.last_data_fetch_date = today
                    user_profile.save()
                else:
                    print("✅ Data already fetched today. Skipping fetch.")

                # 🚀 API key already exists → go to dashboard
                if user_profile.api_key:
                    print("🔑 API Key found. Redirecting to dashboard.")
                    return redirect('dashboard')

            except UserProfile.DoesNotExist:
                # Should not normally happen because of get_or_create
                print("❗ UserProfile not found. Redirecting to Save API Key form.")

            # API key missing → redirect to save_api_key
            print("🚫 No API Key. Redirecting to Save API Key form.")
            return redirect('save_api_key')

        else:
            print("❌ Invalid credentials")
            messages.error(request, "Invalid username or password.")

    else:
        form = AuthenticationForm()

    return render(request, 'auth/login.html', {'form': form})


# Registration view
def register_view(request):
    if request.method == 'POST':
        print("Form submitted!")  # Debugging line
        form = RegisterForm(request.POST)
        if form.is_valid():
            print("Form is valid!")  # Debugging line
            user = form.save(commit=False)
            user.is_active = False  # Activate after email verification
            user.save()

            # Send verification email
            send_verification_email(user, request)

            # Redirect to email verification sent page
            return redirect('email_verification_sent')
        else:
            print("Form is not valid:", form.errors)  # Debugging line
    else:
        form = RegisterForm()

    return render(request, 'auth/register.html', {'form': form})


def send_verification_email(user, request):
    # Generate token and UID
    token = default_token_generator.make_token(user)
    uidb64 = urlsafe_base64_encode(force_bytes(user.pk))

    # Debugging: Print the UID and token
    print(f"UID: {uidb64}, Token: {token}")

    # Construct the verification URL
    domain = get_current_site(request).domain
    verification_link = reverse('activate', kwargs={'uidb64': uidb64, 'token': token})

    full_link = f"http://{domain}{verification_link}"

    # Debugging: Print the full link
    print(f"Verification Link: {full_link}")

    # Create email content
    subject = "Verify your email"
    html_message = render_to_string('auth/email_verification.html', {'link': full_link, 'user': user})
    plain_message = strip_tags(html_message)

    # Send the email
    send_mail(
        subject,
        plain_message,
        EMAIL_HOST_USER,  # Use a valid sender's email
        [user.email],
        html_message=html_message,
    )

    print("Verification email sent.")


def verify_email(request, uidb64, token):
    try:
        # Decode the user ID from the base64 encoded string
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)

        # Debugging: Log the UID and Token
        logger.debug(f"UID: {uidb64}, Token: {token}")

        # Check if the token is valid for the user
        if default_token_generator.check_token(user, token):
            # Mark email as verified and activate the account
            user_profile = getattr(user, 'profile', None)

            if user_profile:
                user_profile.email_verified = True  # Assuming you have an 'email_verified' field in Profile model
                user_profile.save()
            else:
                logger.error(f"User profile not found for user {user.pk}")
                messages.error(request, 'Profile not found for this user.')
                return redirect('register')

            user.is_active = True
            user.save()

            # Show success message
            messages.success(request, 'Email verified successfully! Please log in.')
            return redirect('login')  # Redirect to login after successful email verification
        else:
            # Token is invalid or expired
            messages.error(request, 'Invalid or expired verification link.')
            return redirect('register')

    except (TypeError, ValueError, OverflowError, User.DoesNotExist) as e:
        # Handle cases where the token is invalid, user doesn't exist, or decoding failed
        logger.error(f"Email verification error: {e}")
        messages.error(request, 'Invalid verification link.')
        return redirect('register')


def activate_view(request, uidb64, token):
    try:
        uid = urlsafe_base64_decode(uidb64).decode()
        user = get_user_model().objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, get_user_model().DoesNotExist):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        return redirect('login')
    else:
        return render(request, 'auth/activation_failed.html')


def email_verification_sent_view(request):
    return render(request, 'auth/email_verification_sent.html')


class CustomPasswordResetView(PasswordResetView):
    template_name = 'registration/password_reset_form.html'
    email_template_name = 'registration/password_reset_email.html'
    subject_template_name = 'registration/password_reset_subject.txt'
    success_url = reverse_lazy('password_reset_done')


class CustomPasswordResetDoneView(PasswordResetDoneView):
    template_name = 'registration/password_reset_done.html'


class CustomPasswordResetConfirmView(PasswordResetConfirmView):
    template_name = 'registration/password_reset_confirm.html'
    success_url = reverse_lazy('password_reset_complete')


class CustomPasswordResetCompleteView(PasswordResetCompleteView):
    template_name = 'registration/password_reset_complete.html'


def fetch_and_save_user_data(request):
    try:
        jason_db(request)
        fetch_and_store_stock_data(request)
        print("✅ Data fetching and storing completed successfully.")
    except Exception as e:
        print(f"❌ Error while fetching/saving data: {e}")


@login_required
def save_api_key(request):
    if request.method == 'POST':
        api_key = request.POST.get('api_key')

        if api_key:
            user_profile, created = UserProfile.objects.get_or_create(user=request.user)
            # Debugging: Print to check the UserProfile before saving the API key
            print(f"UserProfile before saving API key: {user_profile.api_key}")

            user_profile.api_key = api_key
            user_profile.save()

            # Send email with the API key
            send_api_key_email(request.user.email, api_key)

            messages.success(request, 'API key saved successfully and emailed to you!')
            # to get all the data
            # jason_db(request)
            # fetch_and_store_stock_data(request)
            # fetch_and_save_user_data(request)

            return redirect('dashboard')  # Redirect to your dashboard
        else:
            messages.error(request, 'API key is required.')

    return render(request, 'auth/api_key_form.html')


def send_api_key_email(user_email, api_key):
    subject = "Your Alpha Vantage API Key"
    message = f"""
    Dear User,

    Thank you for registering with us! 

    Your API key has been saved successfully. Below are your API key details:

    API Key: {api_key}

    Please keep this API key safe and do not share it with anyone. 
    If you have any questions or need further assistance, feel free to contact us.

    Best regards,
    Support Team
    """

    send_mail(
        subject,
        message,
        settings.DEFAULT_FROM_EMAIL,  # Make sure to set this in your settings.py
        [user_email],
        fail_silently=False,
    )


def symbol_selection_stock(request):
    return render(request, 'stocks_analysis/select_symbol_stock.html')


def symbol_selection_news(request):
    return render(request, 'stocks_analysis/select_symbol_news.html')


@login_required
def dashboard_view(request):
    Portfolio.objects.get_or_create(user=request.user)
    return render(request, 'portfolio/dashboard.html')


def logout_view(request):
    logout(request)
    return redirect('login')


def get_stock_file(request):
    if request.method != 'POST':
        # Return an error if the request method is not POST
        return HttpResponseBadRequest("Invalid request method.")

    # Get stock symbol from the form, default to 'TCS.BSE' if not provided
    symbol = request.POST.get('stock_symbol', 'TCS.BSE')

    # Print debug information for symbol
    print(f"Symbol from request: {symbol}")

    # Handle cases where the symbol is missing or None
    if not symbol or symbol.strip().upper() == "NONE":
        return HttpResponseBadRequest("Stock symbol is missing or invalid.")

    try:
        # Fetch and process data for all stocks
        result = fetch_and_store_stock_data(request)

        if len(result) != 10:
            raise ValueError(f"Expected 10 dataframes, got {len(result)}")

        # Unpack the returned data
        (bhartiartl_df, bhartiartl_meta_df,
         icicibank_df, icicibank_meta_df,
         reliance_df, reliance_meta_df,
         tcs_df, tcs_meta_df,
         hdfcbank_df, hdfcbank_meta_df) = result

    except (FileNotFoundError, IndexError, ValueError, TypeError) as e:
        # Log the error (optional, for debugging)
        print(f"Error fetching data: {str(e)}")

        # Render the error page with a context error message
        return render(request, 'portfolio/investment.html', {'error': 'Stock symbol not found or data error.'})

    # Normalize symbol for comparison
    symbol = symbol.strip().upper()

    # Create a dictionary to map symbols to their respective dataframes
    stock_data = {
        "TCS.BSE": (tcs_meta_df, tcs_df),
        "RELIANCE.BSE": (reliance_meta_df, reliance_df),
        "HDFCBANK.BSE": (hdfcbank_meta_df, hdfcbank_df),
        "BHARTIARTL.BSE": (bhartiartl_meta_df, bhartiartl_df),
        "ICICIBANK.BSE": (icicibank_meta_df, icicibank_df),
    }

    # Fetch the correct dataframes based on the symbol
    if symbol in stock_data:
        return stock_data[symbol]
    else:
        # If the symbol doesn't match any known stock, raise an exception
        raise ValueError(f"Unknown stock symbol: {symbol}")


def stock_dataframe(request):
    table1, table2 = get_stock_file(request)
    # table2.drop(columns=['SMA20', 'SMA50', 'Volume_MA'], inplace=True)
    graph_html = generate_graphs(table2)

    df_html1 = table1.to_html(classes='table table-striped')
    df_html2 = table2.to_html(classes='table table-striped')

    context = {
        'df1_html': df_html1,
        'df2_html': df_html2,
        'graph_html': graph_html,
    }

    return render(request, 'stocks_analysis/stock_analysis.html', context)


def investment_platform(request):
    return render(request, 'portfolio/investment.html')


@login_required
def buy_stock(request):
    try:
        portfolio = Portfolio.objects.get(user=request.user)
        available_margin = portfolio.cash_balance
        transactions = Transaction.objects.filter(user=request.user, transaction_type='BUY')
        invested_margin = sum(tx.cost for tx in transactions)
        current_value = Decimal('0.0')

        for tx in transactions:
            try:
                table1, table2 = get_stock_file(request)
                current_stock_price_str = table2['Close'].values[0]
                current_stock_price = Decimal(current_stock_price_str)
                current_value += current_stock_price * tx.quantity
            except (ValueError, TypeError, IndexError):
                return render(request, 'portfolio/buy_stock.html')

        pnl = current_value - invested_margin

        if request.method == 'POST':
            stock_symbol = request.POST.get('stock_symbol', '')
            quantity = int(request.POST.get('quantity', 0))

            if not stock_symbol or quantity <= 0:
                return render(request, 'portfolio/buy_stock.html',
                              {'error': 'Please select a valid stock and quantity.'})

            try:
                table1, table2 = get_stock_file(request)
                stock_price_str = table2['Close'].values[0]
                stock_price = Decimal(stock_price_str)
            except (ValueError, TypeError, IndexError):
                return render(request, 'portfolio/buy_stock.html', {'error': 'Invalid stock price data.'})

            cost = stock_price * quantity

            if portfolio.cash_balance >= cost:
                portfolio.cash_balance -= cost
                portfolio.invested_margin = invested_margin + cost
                portfolio.available_margin = portfolio.cash_balance
                portfolio.current_value = current_value + (stock_price * quantity)
                portfolio.pnl = portfolio.current_value - portfolio.invested_margin
                portfolio.save()

                Transaction.objects.create(
                    user=request.user,
                    stock=stock_symbol,
                    transaction_type='BUY',
                    quantity=quantity,
                    cost=cost,
                    price_at_transaction=stock_price
                )
                messages.success(request, 'Stock purchased successfully! 🎉')
                # Instead of redirect, just send a success flag
                return render(request, 'portfolio/buy_stock.html', {
                    'success': True,
                    'available_margin': portfolio.cash_balance,
                    'invested_margin': portfolio.invested_margin,
                    'current_value': portfolio.current_value,
                    'pnl': portfolio.pnl
                })

            else:
                return render(request, 'portfolio/buy_stock.html', {'error': 'Insufficient funds.'})

        context = {
            'available_margin': available_margin,
            'invested_margin': invested_margin,
            'current_value': current_value,
            'pnl': pnl,
        }
        return render(request, 'portfolio/buy_stock.html', context)

    except Portfolio.DoesNotExist:
        return render(request, 'portfolio/buy_stock.html', {'error': 'Portfolio does not exist.'})


def update_holdings_after_sell(user, stock_symbol, quantity_sold, sale_price):
    """
    Updates the Final_holding model after selling stock.
    :param user: The user selling the stock
    :param stock_symbol: The stock being sold
    :param quantity_sold: Quantity of the stock being sold
    :param sale_price: The price at which the stock is sold
    """

    try:
        # Fetch the user's current holding for the stock
        holding = Final_holding.objects.get(user=user, stock_symbol=stock_symbol)

        # Calculate the new quantity after sale
        new_quantity = holding.quantity - quantity_sold

        if new_quantity > 0:
            # Calculate proportional investment and pnl updates
            proportion_of_investment = Decimal(quantity_sold) / holding.quantity if holding.quantity > 0 else Decimal(
                '0')
            investment_value_sold = holding.investment_value * proportion_of_investment
            pnl_adjustment = (sale_price * quantity_sold) - investment_value_sold

            # Update the holding with reduced quantity, adjusted investment value, and pnl
            holding.quantity = round(new_quantity, 2)
            holding.investment_value = round(holding.investment_value - investment_value_sold, 2)
            holding.current_value = round(holding.current_value - (sale_price * quantity_sold), 2)
            holding.pnl += round(pnl_adjustment, 2)
            holding.save()

        else:
            # If all stock is sold, delete the holding record
            holding.delete()

    except Final_holding.DoesNotExist:
        # Handle case if holding record does not exist (shouldn't happen if we validate quantity before sell)
        print(f"No holding record found for {stock_symbol}.")


@login_required
def sell_stock(request):
    if request.method == 'POST':
        # Retrieve stock symbol and quantity from the form
        stock_symbol = request.POST.get('stock_symbol', 'TCS.BSE')  # Adjust default as needed
        try:
            quantity_to_sell = int(request.POST.get('quantity', 0))
            if quantity_to_sell <= 0:
                raise ValueError("Quantity to sell must be a positive integer.")
        except ValueError:
            return render(request, 'portfolio/sell_stock.html', {'error': 'Invalid quantity to sell.'})

        # Fetch the current stock price from the data source
        try:
            table1, table2 = get_stock_file(request)
            stock_price_str = table2['Close'].values[0]  # Ensure 'Close' column exists
            stock_price = Decimal(stock_price_str)
        except (FileNotFoundError, IndexError, ValueError, TypeError, KeyError):
            return render(request, 'portfolio/sell_stock.html', {'error': 'Stock data unavailable.'})

        # Fetch user's portfolio and transactions for the selected stock
        portfolio = Portfolio.objects.get(user=request.user)
        transactions = Transaction.objects.filter(user=request.user, stock=stock_symbol, transaction_type='BUY')

        # Calculate the total quantity the user owns for this stock
        total_owned_quantity = sum(tx.quantity for tx in transactions)

        # Ensure sufficient quantity to sell
        if total_owned_quantity < quantity_to_sell:
            return render(request, 'portfolio/sell_stock.html', {'error': 'Insufficient stock to sell.'})

        # Calculate sale earnings and proportionate cost of sold stock
        sale_earnings = stock_price * Decimal(quantity_to_sell)
        total_invested = sum(tx.cost for tx in transactions)
        avg_cost_per_share = total_invested / total_owned_quantity if total_owned_quantity > 0 else Decimal(0)
        cost_of_sold_stock = avg_cost_per_share * Decimal(quantity_to_sell)

        # Update portfolio metrics only if they meet conditions
        portfolio.cash_balance += sale_earnings  # Increase cash balance by sale earnings
        portfolio.invested_margin -= cost_of_sold_stock  # Deduct cost from invested margin
        portfolio.pnl += sale_earnings - cost_of_sold_stock  # Calculate and update profit or loss from sale
        portfolio.current_value -= stock_price * Decimal(quantity_to_sell)  # Adjust current value
        portfolio.save()
        # If all stocks of this type are sold, reset cash_balance to default value
        remaining_quantity = total_owned_quantity - quantity_to_sell
        if remaining_quantity == 0:
            portfolio.cash_balance = Decimal('1000000')  # Set to default value (adjust as needed)

        portfolio.save()

        # Record the SELL transaction
        Transaction.objects.create(
            user=request.user,
            stock=stock_symbol,
            transaction_type='SELL',
            quantity=quantity_to_sell,
            cost=cost_of_sold_stock,  # Cost of the shares sold
            price_at_transaction=stock_price  # Price at which the stock is sold
        )
        # After sale transaction logic
        update_holdings_after_sell(request.user, stock_symbol, quantity_to_sell, stock_price)

        # Redirect to the portfolio or dashboard after sale
        return redirect('platform')  # Replace with actual redirect destination

    return render(request, 'portfolio/sell_stock.html')


def get_holdings_as_dataframe(request):
    try:
        # Query Final_holding data for a specific user
        holdings = Final_holding.objects.filter(user=request.user).values(
            'stock_symbol', 'quantity', 'investment_value', 'current_value', 'pnl'
        )

        # Check if holdings is empty
        if not holdings:
            print("No holdings found for this user.")
            return pd.DataFrame()  # Return an empty DataFrame

        # Convert the QuerySet to a DataFrame
        df = pd.DataFrame(list(holdings))

        # Optionally display the DataFrame
        print(df)

        return df

    except ObjectDoesNotExist:
        print("Error: Final_holding data for the user does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame in case of a DB-related error

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame for any other exceptions


@login_required
def investment_view(request):
    # Get or create the user's portfolio and fetch transactions
    portfolio, created = Portfolio.objects.get_or_create(user=request.user)
    transactions = Transaction.objects.filter(user=request.user).order_by('-date')

    # Initialize dictionaries for stock holdings and investment values
    stock_holdings = {}
    stock_investment_value = {}

    # Calculate stock holdings and investment based on transactions
    for tx in transactions:
        stock_symbol = tx.stock

        # Initialize stock in dictionaries if not present
        if stock_symbol not in stock_holdings:
            stock_holdings[stock_symbol] = 0
            stock_investment_value[stock_symbol] = Decimal('0.0')

        # Update holdings and investment value based on transaction type
        if tx.transaction_type == 'BUY':
            stock_holdings[stock_symbol] += tx.quantity
            stock_investment_value[stock_symbol] += tx.quantity * tx.price_at_transaction
        elif tx.transaction_type == 'SELL':
            stock_holdings[stock_symbol] -= tx.quantity

    # Ensure stock quantities are non-negative
    stock_holdings = {stock: max(0, quantity) for stock, quantity in stock_holdings.items()}

    # Initialize totals for portfolio values
    total_investment_value = Decimal('0.0')
    total_current_value = Decimal('0.0')
    total_pnl = Decimal('0.0')

    # Cache for fetched stock prices
    stock_price_cache = {}

    # Calculate current values, investment values, and P&L
    for stock_symbol, quantity in stock_holdings.items():
        if quantity == 0:
            continue  # Skip stocks with zero holdings

        # Get or cache current price
        try:
            if stock_symbol in stock_price_cache:
                current_price = stock_price_cache[stock_symbol]
            else:
                stock_record = CurrentPrice.objects.filter(user=request.user, stock_name=stock_symbol).order_by(
                    '-date').first()
                current_price = Decimal(stock_record.close) if stock_record else Decimal('0.0')
                stock_price_cache[stock_symbol] = current_price

        except Exception as e:
            current_price = Decimal('0.0')
            print(f"Error fetching price for {stock_symbol}: {e}")

        # Calculate the original quantity of buys for the stock
        original_quantity = sum(
            tx.quantity for tx in transactions if tx.stock == stock_symbol and tx.transaction_type == 'BUY'
        )

        # Calculate proportionate investment
        proportion_of_investment = (
                Decimal(quantity) / Decimal(original_quantity)) if original_quantity > 0 else Decimal('0')
        investment_value = stock_investment_value[stock_symbol] * proportion_of_investment
        current_value = quantity * current_price
        pnl = current_value - investment_value

        # Update portfolio totals
        total_investment_value += investment_value
        total_current_value += current_value
        total_pnl += pnl

        # Update holdings in the FinalHolding model
        Final_holding.objects.update_or_create(
            user=request.user,
            stock_symbol=stock_symbol,
            defaults={
                'quantity': round(quantity, 2),
                'investment_value': round(investment_value, 2),
                'current_value': round(current_value, 2),
                'pnl': round(pnl, 2),
            }
        )

    # Fetch total quantity of holdings
    user_holdings = Final_holding.objects.filter(user=request.user)
    total_quantity = user_holdings.aggregate(total_quantity=Sum('quantity'))['total_quantity'] or 0

    # Generate a DataFrame of holdings for visualization
    df = get_holdings_as_dataframe(request)

    # Render graphs
    bar_graph_html = bar_chart_view(df)
    line_graph_html = line_chart_view(df)
    invested_value_pie_graph_html = pie_chart_view(df, 'investment_value', 'Invested Value')
    current_value_pie_graph_html = pie_chart_view(df, 'current_value', 'Current Value')
    pnl_bar_chart_graph_html = pnl_bar_chart_view(df)
    quantity_bar_graph_html = quantity_bar_graph(df)
    percentage_change_pie_graph_html = pie_chart_view(df, 'Percentage_Change', 'Percentage Change of Stocks')

    # Pass context to the template
    context = {
        'portfolio': portfolio,
        'transactions': transactions,
        'created': created,
        'holdings': user_holdings,
        'total_investment_value': round(total_investment_value, 2),
        'total_current_value': round(total_current_value, 2),
        'total_pnl': round(total_pnl, 2),
        'total_quantity': total_quantity,
        'bar_graph_html': bar_graph_html,
        'line_graph_html': line_graph_html,
        'invested_value_pie_graph_html': invested_value_pie_graph_html,
        'current_value_pie_graph_html': current_value_pie_graph_html,
        'pnl_bar_chart_graph_html': pnl_bar_chart_graph_html,
        'quantity_bar_graph_html': quantity_bar_graph_html,
        'percentage_change_pie_graph_html': percentage_change_pie_graph_html,
    }

    return render(request, 'portfolio/platform.html', context)


def transaction_history_view(request):
    transactions = Transaction.objects.filter(user=request.user).order_by('-date')
    context = {
        'transactions': transactions,
    }
    return render(request, 'stocks_analysis/transaction_history.html', context)


@login_required
def jason_db(request):
    function = ["TIME_SERIES_DAILY"]
    symbol = ["TCS.BSE", "RELIANCE.BSE", "HDFCBANK.BSE", "BHARTIARTL.BSE", "ICICIBANK.BSE"]

    # Initialize a list to hold messages
    messages_list = []
    for i in range(len(symbol)):
        # Call the function that fetches, processes, and stores the data
        json_data, error_message = get_daily_time_series(function, symbol[i])
        if error_message:
            messages_list.append(error_message)  # Collect error messages
            continue  # Continue to the next symbol instead of breaking

        # json_data = load_json_from_file('stock_data.json')

        # Extract the stock symbol from the Meta Data
        stock_symbol = json_data["Meta Data"]["2. Symbol"]
        last_refresh = json_data["Meta Data"]["3. Last Refreshed"]
        last_refresh_data = datetime.strptime(last_refresh, '%Y-%m-%d').date()
        meta_data = json_data
        user = request.user

        # Save the data in the database, ensuring to handle updates or duplicates
        StockJason.objects.update_or_create(
            user=user,
            last_refresh_data=last_refresh_data,
            stock_name=stock_symbol,
            defaults={'meta_data': meta_data}
        )

        # Print a success message only if at least one symbol was processed successfully
    if messages_list:
        for msg in messages_list:
            messages.error(request, msg)  # Use Django messages framework for displaying

    print("Stock Data successfully saved to the database.")
    return render(request, 'auth/api_key_form.html', {'message': messages})


@login_required
def fetch_jason_db_data(request):
    # List of stock symbols you want to retrieve
    stock_symbols = ["TCS.BSE", "RELIANCE.BSE", "HDFCBANK.BSE", "BHARTIARTL.BSE", "ICICIBANK.BSE"]

    # Fetch the latest refresh date for each stock symbol for the current user
    latest_records = StockJason.objects.filter(user=request.user, stock_name__in=stock_symbols) \
        .values('stock_name') \
        .annotate(latest_date=Max('last_refresh_data'))  # Get the latest refresh date for each stock

    if latest_records.exists():
        print("Found latest records")
    else:
        print("No latest records found for the user")

    # Create a dictionary to hold the latest stock data
    latest_stock_data = {}

    # Loop through the results to fetch the full stock record
    for record in latest_records:
        stock_name = record['stock_name']
        latest_date = record['latest_date']

        # Get the latest record for each stock symbol
        stock_record = StockJason.objects.filter(user=request.user, stock_name=stock_name,
                                                 last_refresh_data=latest_date).first()

        if stock_record:
            latest_stock_data[stock_name] = stock_record.meta_data  # Store the JSON data

    return latest_stock_data


def df_columns(df):
    if df is None:
        raise ValueError("Received None as DataFrame")
    df.index.name = 'Date'
    df.columns = df.columns.str.replace(r'^\d+\.\s*', '', regex=True).str.strip()
    df.columns = [col.capitalize() for col in df.columns]
    df = df.reset_index()
    df.index = df.index + 1
    return df


def meta_df_columns(df):
    if df is None:
        raise ValueError("Received None as DataFrame")
    df.index.name = 'Parameters'
    df.rename(columns={0: 'Meta data'}, inplace=True)
    # Check if the 'Parameters' column already exists
    if 'Parameters' not in df.columns:
        # Move the index 'Parameters' into a column only if it doesn't exist
        df.reset_index(inplace=True)  # This moves the index to a column named 'Parameters'
    else:
        print("Column 'Parameters' already exists, skipping reset_index.")
    df["Parameters"] = df["Parameters"].astype(str)
    df["Parameters"] = df["Parameters"].str.replace(r'^\d+\.\s*', '', regex=True).str.strip()
    df.index = df.index + 1
    df_transposed = df.set_index('Parameters').T
    df_transposed.reset_index(drop=True)
    df_transposed = df_transposed.reset_index(drop=True)  # ← important fix
    return df_transposed


def fetch_and_process_data_all_stock(request):
    # Fetch the JSON data (assuming `fetch_jason_db_data` retrieves the JSON data)
    json_data = fetch_jason_db_data(request)

    # Convert the JSON data to a DataFrame and reset the index for easier processing
    df = pd.DataFrame.from_dict(json_data).T.reset_index().rename(columns={'index': 'Symbols'})
    df.index = df.index + 1  # Adjust index to start from 1

    # Initialize dictionaries to store the stock data and metadata
    symbol_dfs = {}
    meta_symbol_dfs = {}

    # Process each row in the DataFrame
    for _, row in df.iterrows():
        symbol = row['Symbols']
        time_series_data = row.get('Time Series (Daily)', {})
        meta_data = row.get('Meta Data', {})

        # Check if the time series data and meta data exist
        if not time_series_data or not meta_data:
            print(f"Warning: Missing data for {symbol}")
            continue

        # Process the time series data into a DataFrame
        time_series_df = pd.DataFrame.from_dict(time_series_data, orient='index')
        time_series_df.index = pd.to_datetime(time_series_df.index)
        time_series_df = time_series_df.sort_index(ascending=False)
        symbol_dfs[symbol] = time_series_df

        # Process the metadata into a DataFrame
        meta_df = pd.DataFrame.from_dict(meta_data, orient='index')
        meta_symbol_dfs[symbol] = meta_df

    # Map the symbols to the respective DataFrames
    bhartiartl_df = symbol_dfs.get('BHARTIARTL.BSE')
    hdfcbank_df = symbol_dfs.get('HDFCBANK.BSE')
    icicibank_df = symbol_dfs.get('ICICIBANK.BSE')
    reliance_df = symbol_dfs.get('RELIANCE.BSE')
    tcs_df = symbol_dfs.get('TCS.BSE')

    bhartiartl_meta_df = meta_symbol_dfs.get('BHARTIARTL.BSE')
    hdfcbank_meta_df = meta_symbol_dfs.get('HDFCBANK.BSE')
    icicibank_meta_df = meta_symbol_dfs.get('ICICIBANK.BSE')
    reliance_meta_df = meta_symbol_dfs.get('RELIANCE.BSE')
    tcs_meta_df = meta_symbol_dfs.get('TCS.BSE')

    # Apply additional processing functions
    bhartiartl_df = df_columns(bhartiartl_df)
    icicibank_df = df_columns(icicibank_df)
    reliance_df = df_columns(reliance_df)
    tcs_df = df_columns(tcs_df)
    hdfcbank_df = df_columns(hdfcbank_df)

    bhartiartl_meta_df = meta_df_columns(bhartiartl_meta_df)
    hdfcbank_meta_df = meta_df_columns(hdfcbank_meta_df)
    icicibank_meta_df = meta_df_columns(icicibank_meta_df)
    reliance_meta_df = meta_df_columns(reliance_meta_df)
    tcs_meta_df = meta_df_columns(tcs_meta_df)

    store_stock_data(request, 'BHARTIARTL.BSE', bhartiartl_df)
    store_stock_data(request, 'ICICIBANK.BSE', icicibank_df)
    store_stock_data(request, 'RELIANCE.BSE', reliance_df)
    store_stock_data(request, 'TCS.BSE', tcs_df)
    store_stock_data(request, 'HDFCBANK.BSE', hdfcbank_df)

    return (bhartiartl_df, bhartiartl_meta_df, icicibank_df, icicibank_meta_df,
            reliance_df, reliance_meta_df, tcs_df, tcs_meta_df,
            hdfcbank_df, hdfcbank_meta_df)


def store_stock_data(request, stock_name, stock_df):
    first_row = stock_df.head(1).iloc[0]

    with transaction.atomic():
        # Check for an existing record
        existing_record = CurrentPrice.objects.filter(
            user=request.user,
            stock_name=stock_name,
            date=first_row['Date']
        ).first()

        if existing_record:
            existing_record.open = first_row['Open']
            existing_record.high = first_row['High']
            existing_record.low = first_row['Low']
            existing_record.close = first_row['Close']
            existing_record.volume = first_row['Volume']
            existing_record.save()
            # print(f"Updated the existing record for {stock_name} on {first_row['Date']}")
            return existing_record
        else:
            stock_record = CurrentPrice(
                user=request.user,
                stock_name=stock_name,
                date=first_row['Date'],
                open=first_row['Open'],
                high=first_row['High'],
                low=first_row['Low'],
                close=first_row['Close'],
                volume=first_row['Volume']
            )
            stock_record.save()
            return stock_record


def fetch_and_store_stock_data(request):
    (bhartiartl_df, bhartiartl_meta_df,
     icicibank_df, icicibank_meta_df,
     reliance_df, reliance_meta_df,
     tcs_df, tcs_meta_df,
     hdfcbank_df, hdfcbank_meta_df) = fetch_and_process_data_all_stock(request)

    return bhartiartl_df, bhartiartl_meta_df, icicibank_df, icicibank_meta_df, reliance_df, reliance_meta_df, tcs_df, tcs_meta_df, hdfcbank_df, hdfcbank_meta_df


def about_us(request):
    return render(request, 'auth/aboutus.html')


def aboutt(request):
    return render(request, 'auth/about.html')


def terms_and_conditions(request):
    return render(request, 'auth/termsandconditions.html')


@login_required
def recommendations_view(request):
    # transfer_transactions_to_user_trade()
    # user_trades = UserTrade.objects.filter(user=request.user)
    try:
        df1, df2, df3, df4_unused = predict_df(request)
    except Exception as e:
        print("predict_df error:", e)
        return HttpResponse("Prediction function error", status=500)

    def safe_float(val):
        return None if pd.isna(val) or val == '' else float(val)

    # --- Save df1 to StockPrediction using bulk_create ---
    predictions = []
    for entry in df1.to_dict(orient='records'):
        predictions.append(StockPrediction(
            user=request.user,
            stock_name=entry['Stock_Name'],
            symbol=entry['Symbol'],
            last_refreshed=pd.to_datetime(entry['Last_Refreshed']).date(),
            date=pd.to_datetime(entry['Date']).date(),
            low=safe_float(entry['Low']),
            open=safe_float(entry['Open']),
            high=safe_float(entry['High']),
            close=safe_float(entry['Close']),
            linear_model=safe_float(entry['Linear_Model']),
            lstm_model=safe_float(entry['LSTM_Model']),
            decision_tree_model=safe_float(entry['Decision_Tree_Model']),
            random_forest_model=safe_float(entry['Random_Forest_Model']),
            svm_model=safe_float(entry['SVM_Model'])
        ))
    StockPrediction.objects.bulk_create(predictions, ignore_conflicts=True)

    # --- Save df2 to StockPerformance using bulk_create ---
    performances = []
    for entry in df2.to_dict(orient='records'):
        performances.append(StockPerformance(
            user=request.user,
            stock_name=entry['Stock_Name'],
            symbol=entry['Symbol'],
            linear_model_success_rate=safe_float(entry['Linear_Model_Success_Rate']),
            linear_model_directional_success_rate=safe_float(entry['Linear_Model_Directional_Success_Rate']),
            linear_model_avg_error=safe_float(entry['Linear_Model_Avg_Error']),
            decision_tree_model_success_rate=safe_float(entry['Decision_Tree_Model_Success_Rate']),
            decision_tree_model_directional_success_rate=safe_float(
                entry['Decision_Tree_Model_Directional_Success_Rate']),
            decision_tree_model_avg_error=safe_float(entry['Decision_Tree_Model_Avg_Error']),
            random_forest_model_success_rate=safe_float(entry['Random_Forest_Model_Success_Rate']),
            random_forest_model_directional_success_rate=safe_float(
                entry['Random_Forest_Model_Directional_Success_Rate']),
            random_forest_model_avg_error=safe_float(entry['Random_Forest_Model_Avg_Error']),
            svm_model_success_rate=safe_float(entry['SVM_Model_Success_Rate']),
            svm_model_directional_success_rate=safe_float(entry['SVM_Model_Directional_Success_Rate']),
            svm_model_avg_error=safe_float(entry['SVM_Model_Avg_Error']),
            lstm_model_success_rate=safe_float(entry['LSTM_Model_Success_Rate']),
            lstm_model_directional_success_rate=safe_float(entry['LSTM_Model_Directional_Success_Rate']),
            lstm_model_avg_error=safe_float(entry['LSTM_Model_Avg_Error']),
        ))
    StockPerformance.objects.bulk_create(performances, ignore_conflicts=True)

    # --- Save df3 to BestModelRecord using update_or_create (no bulk_create for this safely) ---
    for entry in df3.to_dict(orient='records'):
        BestModelRecord.objects.update_or_create(
            user=request.user,
            stock_name=entry['Stock_Name'],
            model_name=entry['Model'],
            best_model=entry['Best_Model'],
            defaults={
                'success_rate': safe_float(entry['Success_Rate']),
                'directional_success_rate': safe_float(entry['Directional_Success_Rate']),
                'average_error': safe_float(entry['Average_Error']),
                'normalized_models_score': safe_float(entry['Normalized_Models_Score']),
            }
        )
    # --- Fetch updated data for df2 and df4 for rendering ---
    stock_performance_qs = StockPerformance.objects.filter(user=request.user).order_by('-created')[:5]
    df2 = pd.DataFrame(list(stock_performance_qs.values()))

    best_model_qs = BestModelRecord.objects.filter(user=request.user).order_by('-created')[:25]
    df4 = pd.DataFrame(list(best_model_qs.values()))

    # Get recommendations
    # recommendations = recommend_stocks_total_based(user_trades) if user_trades.exists() else []

    return render(request, 'recommendations/recommendations.html', {
        # 'recommendations': recommendations,
        'df1': df1.to_dict(orient='records'),
        'df2': df2.to_dict(orient='records'),
        'df3': df3.to_dict(orient='records'),
        'df4': df4.to_dict(orient='records'),
    })


def model_explainability_view(request):
    (bhartiartl_df, bhartiartl_meta_df,
     icicibank_df, icicibank_meta_df,
     reliance_df, reliance_meta_df,
     tcs_df, tcs_meta_df,
     hdfcbank_df, hdfcbank_meta_df) = fetch_and_store_stock_data(request)

    stock_data_dict = {
        'Bharti Airtel': bhartiartl_df,
        'ICICI Bank': icicibank_df,
        'Reliance': reliance_df,
        'TCS': tcs_df,
        'HDFC Bank': hdfcbank_df,
    }

    explainability_data = []

    for stock_name, df in stock_data_dict.items():
        # Linear Model
        linear_model, _, _, linear_mse, _, _, _ = train_linear_model(df)
        # linear_model, linear_predictions, actuals, mse, test_indices, df1, lm_df

        # Decision Tree
        decision_model, decision_mse, _ = train_decision_tree_model(df)

        # Random Forest
        rf_model, X_train_rf, X_test_rf, rf_mse, _ = train_random_forest_model(df)
        # rf_predictions = rf_model.predict(X_test_rf)
        # rf_mse = mean_squared_error(df['Close'].iloc[-len(X_test_rf):], rf_predictions)

        # SVM
        svm_model, svm_scaler, X_train_svm_df, X_test_svm_df, mse_svm, _ = train_svm_model(df)

        # LSTM
        lstm_model, _, _, lstm_mse, _ = train_lstm_model(df)

        mse_dict = {
            'Linear_Model': linear_mse,
            'Decision_Tree_Model': decision_mse,
            'Random_Forest_Model': rf_mse,
            'SVM_Model': mse_svm,
            'LSTM_Model': lstm_mse
        }

        best_model = min(mse_dict, key=mse_dict.get)

        # Generate SHAP explainability plots for Linear, Decision Tree, Random Forest, SVM
        summary_imgs = {}
        feature_imgs = {}

        summary_imgs['Linear_Model'], feature_imgs['Linear_Model'] = generate_model_explainability(
            linear_model, X_train_rf, X_test_rf, f"{stock_name}_Linear")

        summary_imgs['Decision_Tree_Model'], feature_imgs['Decision_Tree_Model'] = generate_model_explainability(
            decision_model, X_train_rf, X_test_rf, f"{stock_name}_DT")

        summary_imgs['Random_Forest_Model'], feature_imgs['Random_Forest_Model'] = generate_model_explainability(
            rf_model, X_train_rf, X_test_rf, f"{stock_name}_RF")

        summary_imgs['SVM_Model'], feature_imgs['SVM_Model'] = generate_model_explainability(
            svm_model, X_train_svm_df, X_test_svm_df, f"{stock_name}_SVM")

        # Note: No SHAP for LSTM (Deep learning models require Deep SHAP + TensorFlow backend)

        explainability_data.append({
            'stock_name': stock_name,
            'summary_imgs': summary_imgs,
            'feature_imgs': feature_imgs,
            'best_model_prediction': best_model
        })

    return render(request, 'recommendations/model_explainability.html',
                  {'explainability_data': explainability_data})


def predict_df(request):
    # Fetch stock data and metadata for all stocks
    (bhartiartl_df, bhartiartl_meta_df,
     icicibank_df, icicibank_meta_df,
     reliance_df, reliance_meta_df,
     tcs_df, tcs_meta_df,
     hdfcbank_df, hdfcbank_meta_df) = fetch_and_store_stock_data(request)

    # List of stock dataframes and their corresponding metadata
    stock_dfs = [bhartiartl_df, icicibank_df, reliance_df, tcs_df, hdfcbank_df]
    stock_meta_dfs = [bhartiartl_meta_df, icicibank_meta_df, reliance_meta_df, tcs_meta_df, hdfcbank_meta_df]
    stock_names = ['Bharti Airtel', 'ICICI Bank', 'Reliance', 'TCS', 'HDFC Bank']

    # Initialize an empty list to store merged dataframes
    merged_dfs = []

    # Iterate over all the stock dataframes
    for stock_df, meta_df, stock_name in zip(stock_dfs, stock_meta_dfs, stock_names):

        # Train the models for the current stock
        linear_model, linear_predictions, actuals, mse, test_indices, df1, lm_df = train_linear_model(stock_df)
        decision_tree_model, decision_mse, dt_df = train_decision_tree_model(stock_df)
        random_forest_model, X_train, X_test, rf_mse, rf_df = train_random_forest_model(stock_df)
        svm_model, svm_scaler, X_train_df, X_test_df, mse_svm, svm_df = train_svm_model(stock_df)
        lstm_model, lstm_scaler, lstm_sequence_length, lstm_mse, lstm_df = train_lstm_model(stock_df,
                                                                                            sequence_length=60)

        save_lstm_to_db(request, stock_name, lstm_model, lstm_scaler)
        lstm_modell, lstm_scalerr, sequence_length = load_lstm_from_db(request, stock_name)
        if lstm_modell is None:
            lstm_modell, lstm_scalerr, seq_len = train_lstm_model(stock_df)
            save_lstm_to_db(stock_name, lstm_modell, lstm_scalerr)

        next_day_prediction_lstm_model = predict_next_day_lstm(lstm_modell, lstm_scalerr, stock_df,
                                                               lstm_sequence_length)

        # Predict the next day's value using the trained models
        next_day_prediction_linear_model = predict_next_day(linear_model, stock_df.iloc[1])
        next_day_prediction_decision_tree_model = predict_next_day(decision_tree_model, stock_df.iloc[1])
        next_day_prediction_random_forest_model = predict_next_day(random_forest_model, stock_df.iloc[1])
        next_day_prediction_svm_model = predict_next_day_svm(svm_model, svm_scaler, stock_df.iloc[1])

        success_rate_linear = calculate_success_rate(lm_df, 'Linear_Model')
        directional_success_rate_linear = calculate_directional_success_rate(lm_df, 'Linear_Model')
        avg_error_linear = calculate_avg_error(lm_df, 'Linear_Model')

        success_rate_decision_tree = calculate_success_rate(dt_df, 'Decision_Tree_Model')
        directional_success_rate_decision_tree = calculate_directional_success_rate(dt_df, 'Decision_Tree_Model')
        avg_error_decision_tree = calculate_avg_error(dt_df, 'Decision_Tree_Model')

        success_rate_random_forest = calculate_success_rate(rf_df, 'Random_Forest_Model')
        directional_success_rate_random_forest = calculate_directional_success_rate(rf_df, 'Random_Forest_Model')
        avg_error_random_forest = calculate_avg_error(rf_df, 'Random_Forest_Model')

        success_rate_svm = calculate_success_rate(svm_df, 'SVM_Model')
        directional_success_rate_svm = calculate_directional_success_rate(svm_df, 'SVM_Model')
        avg_error_svm = calculate_avg_error(svm_df, 'SVM_Model')

        success_rate_lstm = calculate_success_rate(lstm_df, 'LSTM_Model')
        directional_success_rate_lstm = calculate_directional_success_rate(lstm_df, 'LSTM_Model')
        avg_error_lstm = calculate_avg_error(lstm_df, 'LSTM_Model')

        # Only keep the necessary columns in the metadata DataFrame
        meta_df = meta_df[['Symbol', 'Last Refreshed']]
        stock_df = stock_df.reset_index()  # Ensure date is in the proper column
        meta_df = meta_df.reset_index(drop=True)  # Reset index to avoid misalignment

        # Extract the last row from the stock data (the most recent data)
        meta_df['Last Refreshed'] = pd.to_datetime(meta_df['Last Refreshed'])

        df1 = stock_df.sort_values(by="Date", ascending=False).iloc[[0]]
        merged_df = pd.merge(meta_df, df1, left_on='Last Refreshed', right_on='Date', how='inner')

        merged_df['Linear_Model'] = next_day_prediction_linear_model
        merged_df['Decision_Tree_Model'] = next_day_prediction_decision_tree_model
        merged_df['Random_Forest_Model'] = next_day_prediction_random_forest_model
        merged_df['SVM_Model'] = next_day_prediction_svm_model
        merged_df['LSTM_Model'] = next_day_prediction_lstm_model
        merged_df['Stock_Name'] = stock_name

        # Add both success rate, directional success rate, and average error columns for each model
        merged_df['Linear_Model_Success_Rate'] = success_rate_linear
        merged_df['Linear_Model_Directional_Success_Rate'] = directional_success_rate_linear
        merged_df['Linear_Model_Avg_Error'] = avg_error_linear

        merged_df['Decision_Tree_Model_Success_Rate'] = success_rate_decision_tree
        merged_df['Decision_Tree_Model_Directional_Success_Rate'] = directional_success_rate_decision_tree
        merged_df['Decision_Tree_Model_Avg_Error'] = avg_error_decision_tree

        merged_df['Random_Forest_Model_Success_Rate'] = success_rate_random_forest
        merged_df['Random_Forest_Model_Directional_Success_Rate'] = directional_success_rate_random_forest
        merged_df['Random_Forest_Model_Avg_Error'] = avg_error_random_forest

        merged_df['SVM_Model_Success_Rate'] = success_rate_svm
        merged_df['SVM_Model_Directional_Success_Rate'] = directional_success_rate_svm
        merged_df['SVM_Model_Avg_Error'] = avg_error_svm

        merged_df['LSTM_Model_Success_Rate'] = success_rate_lstm
        merged_df['LSTM_Model_Directional_Success_Rate'] = directional_success_rate_lstm
        merged_df['LSTM_Model_Avg_Error'] = avg_error_lstm

        # Append the merged dataframe to the list
        merged_dfs.append(merged_df)

    # Concatenate all merged dataframes into one dataframe
    final_merged_df = pd.concat(merged_dfs, ignore_index=True)
    final_merged_df.drop(columns=['Volume', 'Target'], inplace=True)
    # Reorder columns to place 'Stock Name' first
    final_merged_df = final_merged_df[['Stock_Name'] + [col for col in final_merged_df.columns if col != 'Stock_Name']]
    final_merged_df.rename(columns={'Last Refreshed': 'Last_Refreshed'}, inplace=True)
    # Round selected columns
    round_columns = [
        'Low', 'Open', 'High', 'Close',
        'Linear_Model',
        'Decision_Tree_Model',
        'Random_Forest_Model',
        'SVM_Model',
        'LSTM_Model'
    ]
    final_merged_df[round_columns] = final_merged_df[round_columns].round(2)
    # Drop the 'index' column
    final_merged_df = final_merged_df.drop(columns=['index'])

    df1 = final_merged_df.loc[:, 'Stock_Name':'SVM_Model']
    df2 = final_merged_df[['Stock_Name', 'Symbol'] + list(final_merged_df.loc[:, 'Linear_Model_Success_Rate':].columns)]

    # Calculate scores for each model and get the best one
    df3, df4 = calculate_model_scores(final_merged_df)

    return df1, df2, df3, df4


from django.shortcuts import render
from .models import StockPrediction
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot


def generate_stock_visualizations_view(request):
    records = StockPrediction.objects.all().values(
        'stock_name', 'symbol', 'last_refreshed', 'date', 'low', 'open', 'high', 'close',
        'linear_model', 'lstm_model', 'decision_tree_model', 'random_forest_model'
    )

    if not records:
        return render(request, 'visualizations/no_data.html')

    df = pd.DataFrame(records)
    all_charts = []

    for stock in df['stock_name'].unique():
        df_stock = df[df['stock_name'] == stock].iloc[0]

        # Bar chart
        bar_fig = go.Figure(data=[
            go.Bar(name='Actual Close', x=['Actual'], y=[df_stock['close']]),
            go.Bar(name='Linear', x=['Linear'], y=[df_stock['linear_model']]),
            go.Bar(name='LSTM', x=['LSTM'], y=[df_stock['lstm_model']]),
            go.Bar(name='Decision Tree', x=['Decision Tree'], y=[df_stock['decision_tree_model']]),
            go.Bar(name='Random Forest', x=['Random Forest'], y=[df_stock['random_forest_model']])
        ])
        bar_fig.update_layout(
            title=f"{stock} - One Day Prediction Comparison",
            xaxis_title="Model",
            yaxis_title="Price",
            barmode='group',
            template='plotly_white'
        )
        bar_div = plot(bar_fig, output_type='div')

        all_charts.append({
            'stock_name': stock,
            'chart': bar_div,
            'linear_model': df_stock['linear_model'],
            'lstm_model': df_stock['lstm_model'],
            'decision_tree_model': df_stock['decision_tree_model'],
            'random_forest_model': df_stock['random_forest_model'],
            'actual_close': df_stock['close']
        })

    return render(request, 'recommendations/predictions_graphs.html', {
        'all_charts': all_charts
    })


from django.shortcuts import render
from auth_app.forms import HyperparameterForm
from auth_app.aiml.custom_model import train_test_data, \
    linear_model_hyper_tuning_chart, decision_tree_hyper_tuning_chart, random_forest_hyper_tuning_chart, \
    svm_hyper_tuning_chart, lstm_hyper_tuning_chart


# train_decision_tree_hyper_tuning, train_random_forest_hyper_tuning, train_svm_hyper_tuning, train_lstm_hyper_tuning, \
# save_predictions_to_db_hyper_tuning

def train_models_with_hyperparameters(request, hyperparams):
    symbol = hyperparams['stock_symbol']
    # Extract linear specific hyperparameters
    fit_intercept = str(hyperparams['fit_intercept']).lower() == 'true'
    regularization_type = hyperparams['regularization_type']
    alpha = float(hyperparams['alpha'])

    # Extract decision tree specific hyperparameters

    # min_samples_split = int(hyperparams.get('min_samples_split', 2))  # Default to 2
    # min_samples_split = int(hyperparams.get('min_samples_split') or 2)
    criterion = hyperparams['criterion']
    raw_max_depth = hyperparams.get('max_depth')
    raw_min_samples_split = hyperparams.get('min_samples_split')
    max_depth = int(raw_max_depth) if raw_max_depth not in [None, '', 'None'] else None
    min_samples_split = int(raw_min_samples_split) if raw_min_samples_split not in [None, '', 'None'] else None

    # max_depth = int(hyperparams.get('max_depth')) if hyperparams.get('max_depth') and hyperparams.get(
    #     'max_depth').isdigit() else 5

    # Extract random forest specific hyperparameters
    criterion_rf = hyperparams['criterion_rf']
    min_samples_split_rf = int(hyperparams.get('min_samples_split_rf', 2))  # Default to 2
    n_estimators = int(hyperparams.get('n_estimators', 100))
    rf_max_depth = int(hyperparams.get('rf_max_depth')) if hyperparams.get('rf_max_depth') and hyperparams.get(
        'rf_max_depth').isdigit() else 5

    # Extract SVM specific hyperparameters

    kernel = hyperparams.get('kernel', 'rbf')
    C = float(hyperparams.get('C', 1.0))
    epsilon = float(hyperparams.get('epsilon', 0.1))
    gamma = hyperparams.get('gamma', 'scale')  # can be 'scale', 'auto', or float
    degree = int(hyperparams.get('degree', 3))
    coef0 = float(hyperparams.get('coef0', 0.0))

    # Extract LSTM specific hyperparameters

    lstm_units = int(hyperparams.get('lstm_units', 32))
    epochs = int(hyperparams.get('epochs', 20))
    batch_size = int(hyperparams.get('batch_size', 32))
    num_layers = int(hyperparams.get('num_layers', 1))  # from dropdown

    learning_rate = float(hyperparams.get('learning_rate', 0.001))  # from dropdown (stringified)
    dropout = float(hyperparams.get('dropout', 0.2))  # from dropdown (stringified)

    optimizer = hyperparams.get('optimizer', 'adam')
    loss_function = hyperparams.get('loss_function', 'mse')
    activation_function = hyperparams.get('activation_function', 'linear')

    # Fetch stock data using request
    (bhartiartl_df, bhartiartl_meta_df,
     icicibank_df, icicibank_meta_df,
     reliance_df, reliance_meta_df,
     tcs_df, tcs_meta_df,
     hdfcbank_df, hdfcbank_meta_df) = fetch_and_store_stock_data(request)

    symbol_to_df = {
        'BHARTIARTL.BSE': bhartiartl_df,
        'ICICIBANK.BSE': icicibank_df,
        'RELIANCE.BSE': reliance_df,
        'TCS.BSE': tcs_df,
        'HDFCBANK.BSE': hdfcbank_df,
    }

    df = symbol_to_df.get(symbol)

    if df is None or df.empty:
        return "<p>No data available for the selected symbol.</p>"
    latest_record = df.to_dict(orient='records')[0]
    # Accumulate all summaries in a list
    all_summaries = []
    X_train, X_test, y_train, y_test = train_test_data(df)

    chart_html, summary_lm_df = linear_model_hyper_tuning_chart(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, stock_name=symbol,
        regularization=regularization_type,
        alpha=alpha, fit_intercept=fit_intercept
    )
    # print(summary_lm_df.to_string())

    dt_chart, dt_importance_chart, summary_dt_df = decision_tree_hyper_tuning_chart(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, stock_name=symbol,
        criterion=criterion,
        min_samples_split=min_samples_split,
        max_depth=max_depth
    )

    main_chart, importance_chart, summary_rf_df = random_forest_hyper_tuning_chart(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, stock_name=symbol,
        criterion=criterion_rf,
        min_samples_split=min_samples_split_rf,
        n_estimators=n_estimators,
        rf_max_depth=rf_max_depth
    )

    main_chart_svm, importance_chart_svm, summary_svm_df = svm_hyper_tuning_chart(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, stock_name=symbol,
        kernel=kernel,
        C=C,
        epsilon=epsilon,
        gamma=gamma,
        degree=degree,
        coef0=coef0

    )
    main_chart_lstm, summary_lstm_df = lstm_hyper_tuning_chart(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, stock_name=symbol,
        lstm_units=lstm_units,
        epochs=epochs,
        batch_size=batch_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        dropout=dropout,
        optimizer=optimizer,
        loss_function=loss_function,
        activation_function=activation_function

    )
    # After generating summary DataFrames
    all_summaries.append(summary_lm_df)
    all_summaries.append(summary_dt_df)
    all_summaries.append(summary_rf_df)
    all_summaries.append(summary_svm_df)
    all_summaries.append(summary_lstm_df)
    # Combine all summaries into one DataFrame
    final_summary_df = pd.concat(all_summaries, ignore_index=True)

    # final_summary_df['Best_Model'] = final_summary_df['R2'] == final_summary_df['R2'].max()
    final_summary_df['Best_Model'] = final_summary_df['R2'].apply(
        lambda x: "Yes" if x == final_summary_df['R2'].max() else "No")

    summary_df = final_summary_df.round(3).to_dict(orient='records')
    # print(summary_df.to_string())

    return chart_html, dt_chart, dt_importance_chart, main_chart, importance_chart, main_chart_svm, importance_chart_svm, main_chart_lstm, summary_df, latest_record


def hyperparameter_training_view(request):
    chart_html = None
    dt_chart = None
    dt_importance_chart = None
    main_chart = None
    importance_chart = None
    main_chart_svm = None
    importance_chart_svm = None
    main_chart_lstm = None
    summary_df = None
    latest_record = None

    if request.method == 'POST':
        form = HyperparameterForm(request.POST)
        if form.is_valid():
            hyperparams = form.cleaned_data
            chart_html, dt_chart, dt_importance_chart, main_chart, importance_chart, main_chart_svm, importance_chart_svm, main_chart_lstm, summary_df, latest_record = train_models_with_hyperparameters(
                request,
                hyperparams)  # Custom logic
    else:
        form = HyperparameterForm()

    return render(request, 'recommendations/train_with_hyperparams.html',
                  {'form': form,
                   'chart_html': chart_html,
                   'dt_chart': dt_chart,
                   'dt_importance_chart': dt_importance_chart,
                   'main_chart': main_chart,
                   'importance_chart': importance_chart,
                   'main_chart_svm': main_chart_svm,
                   'importance_chart_svm': importance_chart_svm,
                   'main_chart_lstm': main_chart_lstm,
                   'summary_df': summary_df,
                   'latest_record': latest_record
                   })

# from django.shortcuts import render
# from .forms import HyperparameterForm
# from .models import StockPrediction
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
#
# def train_model_view(request):
#     form = HyperparameterForm(request.POST or None)
#     if form.is_valid():
#         # Get the selected stock symbol and hyperparameters from the form
#         symbol = form.cleaned_data['symbol']
#         fit_intercept = form.cleaned_data['fit_intercept']
#         max_depth = form.cleaned_data['max_depth']
#         criterion = form.cleaned_data['criterion']
#         n_estimators = form.cleaned_data['n_estimators']
#         rf_max_depth = form.cleaned_data['rf_max_depth']
#         kernel = form.cleaned_data['kernel']
#         C = form.cleaned_data['C']
#         lstm_units = form.cleaned_data['lstm_units']
#         epochs = form.cleaned_data['epochs']
#
#         # Retrieve the stock data based on the selected symbol
#         (bhartiartl_df, bhartiartl_meta_df,
#          icicibank_df, icicibank_meta_df,
#          reliance_df, reliance_meta_df,
#          tcs_df, tcs_meta_df,
#          hdfcbank_df, hdfcbank_meta_df) = fetch_and_store_stock_data(request)
#         symbol_map = {
#             "BHARTIARTL.BSE": bhartiartl_df,
#             "ICICIBANK.BSE": icicibank_df,
#             "RELIANCE.BSE": reliance_df,
#             "TCS.BSE": tcs_df,
#             "HDFCBANK.BSE": hdfcbank_df
#         }
#         df = symbol_map.get(symbol, bhartiartl_df)  # default to bhartiartl_df if not found
#
#         # Train each model and collect results
#         linear_model, linear_predictions, y_test, lm_mse, _, df1, df_with_predictions = train_linear_model_hyper_tuning(df, fit_intercept)
#         # decision_tree_model, dt_predictions, dt_mse = train_decision_tree_hyper_tuning(df, max_depth, criterion)
#         # random_forest_model, rf_predictions, rf_mse = train_random_forest_hyper_tuning(df, n_estimators, rf_max_depth)
#         # svm_model, svm_predictions, svm_mse = train_svm_hyper_tuning(df, kernel, C)
#         # lstm_model, lstm_predictions, lstm_mse = train_lstm_hyper_tuning(df, lstm_units, epochs)
#
#         # Save predictions in the database
#         # save_predictions_to_db_hyper_tuning(symbol, linear_predictions, dt_predictions, rf_predictions, svm_predictions, lstm_predictions)
#
#         # Prepare the chart or result to display
#         chart_html = create_chart(df_with_predictions)
#
#         return render(request, 'Hyperparameter_model_training.html', {
#             'form': form,
#             'chart_html': chart_html
#         })
#
#     return render(request, 'recommendations/Hyperparameter_model_training.html', {'form': form})
