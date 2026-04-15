r"""
Anomaly Injector — creates error and slow traces in the demo app.

Sends requests designed to trigger:
  - Checkout with invalid data → error traces
  - Rapid product browsing → fast routine traces (should be deprioritized)
  - Cart operations → moderate traces
  - Currency conversions → fast traces

Usage:
  cd D:\IIT\...\fsbs-platform
  python validation\anomaly_injector.py
"""

import time
import random
import logging
import requests
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [injector] %(levelname)s: %(message)s',
)
logger = logging.getLogger('injector')

FRONTEND_URL = "http://localhost:8080"

# Product IDs from the demo app
PRODUCT_IDS = [
    "OLJCESPC7Z",  # Vintage Typewriter
    "66VCHSJNUP",  # Vintage Camera Lens
    "1YMWWN1N4O",  # Home Barista Kit
    "L9ECAV7KIM",  # Terrarium
    "2ZYFJ3GM2N",  # Film Camera
    "0PUK6V6EV0",  # Vintage Record Player
    "LS4PSXUNUM",  # Metal Camping Mug
    "9SIQT8TOJO",  # City Bike
    "6E92ZMYYFZ",  # Air Plant
]

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "TRY"]


class AnomalyInjector:
    """Generates diverse traffic patterns including anomalies."""

    def __init__(self):
        self.session = requests.Session()
        self.stats = {
            'browse_ok': 0,
            'browse_err': 0,
            'cart_ok': 0,
            'cart_err': 0,
            'checkout_ok': 0,
            'checkout_err': 0,
            'total_requests': 0,
        }

    def browse_product(self, product_id: str) -> bool:
        """Browse a product page — generates a multi-service trace."""
        try:
            resp = self.session.get(
                f"{FRONTEND_URL}/product/{product_id}",
                timeout=10,
            )
            self.stats['total_requests'] += 1
            if resp.status_code == 200:
                self.stats['browse_ok'] += 1
                return True
            else:
                self.stats['browse_err'] += 1
                return False
        except Exception as e:
            self.stats['browse_err'] += 1
            self.stats['total_requests'] += 1
            return False

    def add_to_cart(self, product_id: str, quantity: int = 1) -> bool:
        """Add item to cart — generates cart + product catalog traces."""
        try:
            resp = self.session.post(
                f"{FRONTEND_URL}/cart",
                data={
                    'product_id': product_id,
                    'quantity': quantity,
                },
                timeout=10,
                allow_redirects=True,
            )
            self.stats['total_requests'] += 1
            if resp.status_code == 200:
                self.stats['cart_ok'] += 1
                return True
            else:
                self.stats['cart_err'] += 1
                return False
        except Exception:
            self.stats['cart_err'] += 1
            self.stats['total_requests'] += 1
            return False

    def checkout(self, expect_error: bool = False) -> bool:
        """
        Attempt checkout.
        
        With valid data → success trace (multi-service: checkout+payment+shipping+email)
        With invalid data → may produce error spans
        """
        if expect_error:
            # Invalid credit card data to potentially trigger errors
            checkout_data = {
                'email': 'test@error.invalid',
                'street_address': '',
                'zip_code': '00000',
                'city': '',
                'state': '',
                'country': '',
                'credit_card_number': '0000000000000000',
                'credit_card_expiration_month': '1',
                'credit_card_expiration_year': '2020',  # expired
                'credit_card_cvv': '000',
            }
        else:
            checkout_data = {
                'email': f'user{random.randint(1,9999)}@example.com',
                'street_address': '123 Test St',
                'zip_code': '10001',
                'city': 'New York',
                'state': 'NY',
                'country': 'US',
                'credit_card_number': '4432801561520454',
                'credit_card_expiration_month': '1',
                'credit_card_expiration_year': '2030',
                'credit_card_cvv': '672',
            }

        try:
            resp = self.session.post(
                f"{FRONTEND_URL}/cart/checkout",
                data=checkout_data,
                timeout=15,
                allow_redirects=True,
            )
            self.stats['total_requests'] += 1
            if resp.status_code == 200:
                self.stats['checkout_ok'] += 1
                return True
            else:
                self.stats['checkout_err'] += 1
                return False
        except Exception:
            self.stats['checkout_err'] += 1
            self.stats['total_requests'] += 1
            return False

    def set_currency(self, currency: str) -> bool:
        """Change currency — generates currency service traces."""
        try:
            resp = self.session.post(
                f"{FRONTEND_URL}/setCurrency",
                data={'currency_code': currency},
                timeout=10,
                allow_redirects=True,
            )
            self.stats['total_requests'] += 1
            return resp.status_code == 200
        except Exception:
            self.stats['total_requests'] += 1
            return False

    def empty_cart(self) -> bool:
        """Empty the cart."""
        try:
            resp = self.session.post(
                f"{FRONTEND_URL}/cart/empty",
                timeout=10,
                allow_redirects=True,
            )
            self.stats['total_requests'] += 1
            return resp.status_code == 200
        except Exception:
            self.stats['total_requests'] += 1
            return False


def run_normal_traffic(injector: AnomalyInjector, count: int = 20):
    """Generate routine browsing traffic."""
    logger.info(f"Generating {count} routine browse requests...")
    for i in range(count):
        product = random.choice(PRODUCT_IDS)
        injector.browse_product(product)
        time.sleep(0.2)


def run_checkout_traffic(injector: AnomalyInjector, count: int = 5):
    """Generate checkout traffic (multi-service, higher value)."""
    logger.info(f"Generating {count} checkout flows...")
    for i in range(count):
        # Browse → Add to cart → Checkout
        product = random.choice(PRODUCT_IDS)
        injector.browse_product(product)
        injector.add_to_cart(product, quantity=random.randint(1, 3))
        injector.checkout(expect_error=False)
        injector.empty_cart()
        time.sleep(0.5)


def run_error_burst(injector: AnomalyInjector, count: int = 10):
    """Generate a burst of potentially erroneous requests."""
    logger.info(f"Generating {count} error-prone requests...")
    for i in range(count):
        # Try invalid checkouts
        product = random.choice(PRODUCT_IDS)
        injector.add_to_cart(product)
        injector.checkout(expect_error=True)
        injector.empty_cart()

        # Try browsing non-existent products
        injector.browse_product("INVALID_PRODUCT_ID_" + str(i))

        # Rapid currency switches (stress test)
        injector.set_currency(random.choice(CURRENCIES))
        time.sleep(0.1)

def run_slow_traffic(injector: AnomalyInjector, count: int = 10):
    """Generate intentionally slow requests."""
    logger.info(f"Generating {count} slow requests...")
    for i in range(count):
        # Browse many products in sequence to create long traces
        for _ in range(5):
            product = random.choice(PRODUCT_IDS)
            injector.browse_product(product)
            time.sleep(0.5)  # Small delay between requests
        
        # Add multiple items to cart
        for _ in range(3):
            product = random.choice(PRODUCT_IDS)
            injector.add_to_cart(product, quantity=random.randint(1, 5))
            time.sleep(0.3)
        
        time.sleep(1)

def run_mixed_traffic(injector: AnomalyInjector, duration_minutes: float = 5):
    """
    Run a realistic mix of traffic for the specified duration.
    
    Traffic mix:
      - 50% routine browsing (low value)
      - 20% checkout flows (high value) 
      - 20% slow traces (high value)      ← UPDATED
      - 10% error-prone requests (highest value)  ← UPDATED
    """
    end_time = time.time() + (duration_minutes * 60)
    cycle = 0

    while time.time() < end_time:
        cycle += 1
        elapsed = duration_minutes - (end_time - time.time()) / 60
        logger.info(
            f"── Traffic cycle {cycle} "
            f"({elapsed:.1f}/{duration_minutes:.1f} min) ──"
        )

        # 50% routine browsing
        run_normal_traffic(injector, count=10)  # ← REDUCED from 12

        # 20% checkout flows
        run_checkout_traffic(injector, count=3)

        # 20% slow traces (every cycle)  ← NEW
        run_slow_traffic(injector, count=4)

        # 10% errors (every other cycle to create bursts)
        if cycle % 2 == 0:
            run_error_burst(injector, count=5)  # ← KEPT same

        # Log stats
        s = injector.stats
        logger.info(
            f"Stats: total={s['total_requests']} | "
            f"browse={s['browse_ok']}/{s['browse_ok']+s['browse_err']} | "
            f"cart={s['cart_ok']}/{s['cart_ok']+s['cart_err']} | "
            f"checkout={s['checkout_ok']}/{s['checkout_ok']+s['checkout_err']}"
        )

        time.sleep(2)

    logger.info("Traffic generation complete!")
    logger.info(f"Final stats: {injector.stats}")

def main():
    logger.info("=" * 60)
    logger.info("FSBS Anomaly Injector")
    logger.info(f"Target: {FRONTEND_URL}")
    logger.info("=" * 60)

    # Verify frontend is accessible
    try:
        resp = requests.get(f"{FRONTEND_URL}/", timeout=15)
        logger.info(f"Frontend accessible: HTTP {resp.status_code}")
    except Exception as e:
        logger.error(f"Cannot reach frontend: {e}")
        logger.error("Make sure docker compose is running!")
        return

    injector = AnomalyInjector()

    logger.info("Starting 5-minute mixed traffic generation...")
    logger.info("  60% routine browsing")
    logger.info("  25% checkout flows")
    logger.info("  15% error-prone requests")
    logger.info("")

    run_mixed_traffic(injector, duration_minutes=5)


if __name__ == '__main__':
    main()