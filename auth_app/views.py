from datetime import datetime
from decimal import Decimal
import pandas as pd
from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.db.models import Max, Sum
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout
from auth_project.settings import EMAIL_HOST_USER
from .custom_utils.fetching import get_daily_time_series
from .custom_utils.graphs import generate_graphs, bar_chart_view, pie_chart_view, line_chart_view, quantity_bar_graph, \
    pnl_bar_chart_view
from .forms import RegisterForm
from .middlewares import auth, guest
from .models import Portfolio, Transaction, UserProfile, Final_holding, StockJason, CurrentPrice
from django.contrib.auth import login
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.urls import reverse, reverse_lazy
from django.contrib.auth.views import PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, \
    PasswordResetCompleteView
from django.contrib.auth import get_user_model
from .forms import AlphaVantageRegistrationForm
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail
from django.conf import settings
from django.utils.http import urlsafe_base64_decode
from django.utils.encoding import force_str
from django.contrib.auth.tokens import default_token_generator
from django.contrib import messages
from django.shortcuts import redirect
from django.contrib.auth.models import User
from django.http import HttpResponseBadRequest
from django.shortcuts import render

import logging

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


# def login_view(request):
#     if request.method == 'POST':
#
#         form = AuthenticationForm(request, data=request.POST)
#         if form.is_valid():
#             user = form.get_user()
#             print(f"✅ User '{user.username}' logged in Successfully ")
#
#             login(request, user)
#
#             # 👇 Check if API key exists in UserProfile
#             try:
#                 user_profile = UserProfile.objects.get(user=user)
#                 if user_profile.api_key:
#                     print("🔑 API Key found. Redirecting to dashboard.")
#                     return redirect('dashboard')
#             except UserProfile.DoesNotExist:
#                 pass
#
#             # API key missing → redirect to save_api_key
#             print("🚫 No API Key. Redirecting to Save API Key form.")
#             return redirect('save_api_key')
#
#         else:
#             print("❌ Invalid credentials")
#             messages.error(request, "Invalid username or password.")
#
#     else:
#         form = AuthenticationForm()
#         print("Authentication failed")
#     return render(request, 'auth/login.html', {'form': form})

from django.utils.timezone import now
from .models import UserProfile


def login_view(request):
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
        print("Authentication failed")

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


def alpha_vantage_register(request):
    if request.method == 'POST':
        form = AlphaVantageRegistrationForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            # Here, you would typically send a request to register the user
            # But since Alpha Vantage requires manual registration, we simulate this
            messages.success(request, 'Registration successful! Please check your email for the API key.')
            # Redirect or render the same page with success message
            return redirect('alpha_vantage_register')
    else:
        form = AlphaVantageRegistrationForm()
    return render(request, 'auth/register_alpha_vantage.html', {'form': form})


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


def save_api_key(request):
    if request.method == 'POST':
        api_key = request.POST.get('api_key')

        if api_key:
            user_profile, created = UserProfile.objects.get_or_create(user=request.user)
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


@auth
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


# def stock_dataframe(request):
#     # Get the data and generate the HTML components
#     table1, table2 = get_stock_file(request)
#     graph_html = generate_graphs(table2)
#     df_html1 = table1.to_html(classes='table table-striped')
#     df_html2 = table2.to_html(classes='table table-striped')
#
#     # Check if we are in 'graph' view mode
#     view_mode = request.GET.get('view', 'full')  # Default to 'full' view
#
#     context = {
#         'df1_html': df_html1,
#         'df2_html': df_html2,
#         'graph_html': graph_html,
#         'view_mode': view_mode,
#     }
#
#     return render(request, 'stocks_analysis/stock_analysis.html', context)


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


# def display_graph(request):
#     # Generate the graph HTML
#     stock_data = get_stock_file(request)
#     # Check if stock_data is an HttpResponse (indicating an error was returned)
#     if isinstance(stock_data, HttpResponse):
#         # Return the error response directly
#         return stock_data
#     #
#     #     # Unpack the returned data
#     table1, table2 = stock_data
#     #     # table1, table2 = get_stock_file(request)
#     #
#     graph_html = generate_graphs(table2)
#
#     # Pass the graph HTML to the template context
#     context = {'graph_html': graph_html}
#
#     return render(request, 'stocks_analysis/display_graph.html', context)


def investment_platform(request):
    return render(request, 'portfolio/investment.html')


def lets_get_started(request):
    return render(request, 'portfolio/dashboard.html')


@login_required
def buy_stock(request):
    try:
        # Get the user's portfolio
        portfolio = Portfolio.objects.get(user=request.user)

        # Available margin (cash balance)
        available_margin = portfolio.cash_balance

        # Get all BUY transactions for the user
        transactions = Transaction.objects.filter(user=request.user, transaction_type='BUY')

        # Calculate Invested Margin (sum of all BUY transaction costs)
        invested_margin = sum(tx.cost for tx in transactions)

        # Calculate the current value of all stocks in the user's portfolio
        current_value = Decimal('0.0')
        for tx in transactions:
            try:

                table1, table2 = get_stock_file(request)

                # Get the latest closing price
                current_stock_price_str = table2['Close'].values[0]
                current_stock_price = Decimal(current_stock_price_str)

                # Calculate the current value for this stock (quantity * current price)
                current_value += current_stock_price * tx.quantity

            except (ValueError, TypeError, IndexError):
                # Handle any issues with fetching or parsing stock price data
                return render(request, 'portfolio/buy_stock.html')

                # Calculate the P&L (Profit and Loss)
        pnl = current_value - invested_margin

        if request.method == 'POST':
            # Get stock symbol and quantity from the form
            stock_symbol = request.POST.get('stock_symbol', 'TCS.BSE')
            quantity = int(request.POST.get('quantity', 0))  # Get the quantity to buy

            try:

                table1, table2 = get_stock_file(request)
                stock_price_str = table2['Close'].values[0]
                stock_price = Decimal(stock_price_str)

            except (ValueError, TypeError, IndexError):
                # Handle any issues with fetching or converting the stock price
                return render(request, 'portfolio/buy_stock.html', {'error': 'Invalid stock price data.'})

            # Calculate the total cost of the stock purchase
            cost = stock_price * quantity

            if portfolio.cash_balance >= cost:
                # Update portfolio's cash balance and other fields
                portfolio.cash_balance -= cost
                portfolio.invested_margin = invested_margin + cost
                portfolio.available_margin = portfolio.cash_balance  # Update available margin after purchase
                portfolio.current_value = current_value + (
                        stock_price * quantity)  # Update current value after purchase
                portfolio.pnl = portfolio.current_value - portfolio.invested_margin  # Recalculate P&L
                portfolio.save()

                # Record the transaction
                Transaction.objects.create(
                    user=request.user,
                    stock=stock_symbol,
                    transaction_type='BUY',
                    quantity=quantity,
                    cost=cost,
                    price_at_transaction=stock_price
                )

                return redirect('platform')
            else:
                # Insufficient balance to make the purchase
                return render(request, 'portfolio/buy_stock.html', {'error': 'Insufficient funds.'})

        context = {
            'available_margin': available_margin,
            'invested_margin': invested_margin,
            'current_value': current_value,  # Display the current portfolio value
            'pnl': pnl,  # Display the updated P&L
        }

        return render(request, 'portfolio/buy_stock.html', context)

    except Portfolio.DoesNotExist:
        return render(request, 'portfolio/buy_stock.html', {'error': 'Portfolio does not exist.'})


# Assume this is the helper to retrieve stock data


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


# def save_json_to_file(json_data, filename, directory='C:/Users/Ankush/PycharmProjects/Django/auth_app/files'):
#     # Ensure the directory exists
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
#     # Create the full file path
#     file_path = os.path.join(directory, filename)
#
#     # Save json_data to a file
#     with open(file_path, 'w') as json_file:
#         json.dump(json_data, json_file, indent=4)
#
#     print(f"Data saved to {file_path}")


# def load_json_from_file(filename, directory='C:/Users/Ankush/PycharmProjects/Django/auth_app/files'):
#     # Create the full file path
#     file_path = os.path.join(directory, filename)
#
#     # Load the JSON data from the file
#     with open(file_path, 'r') as json_file:
#         json_data = json.load(json_file)
#
#     print(f"Data loaded from {file_path}")
#     return json_data


# Assuming the JSON file path is given, and you want to handle this in a view
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
    df.index.name = 'Date'
    df.columns = df.columns.str.replace(r'^\d+\.\s*', '', regex=True).str.strip()
    df.columns = [col.capitalize() for col in df.columns]
    df = df.reset_index()
    df.index = df.index + 1
    return df


def meta_df_columns(df):
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
            # print(f"Created a new record for {stock_name} on {first_row['Date']}")
            return stock_record


def fetch_and_store_stock_data(request):
    (bhartiartl_df, bhartiartl_meta_df,
     icicibank_df, icicibank_meta_df,
     reliance_df, reliance_meta_df,
     tcs_df, tcs_meta_df,
     hdfcbank_df, hdfcbank_meta_df) = fetch_and_process_data_all_stock(request)

    return bhartiartl_df, bhartiartl_meta_df, icicibank_df, icicibank_meta_df, reliance_df, reliance_meta_df, tcs_df, tcs_meta_df, hdfcbank_df, hdfcbank_meta_df
