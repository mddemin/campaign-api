#!/usr/bin/env python3
"""Revcontent Campaign API Integration.

Developer Test Assignment

This module demonstrates:
    1. Authentication with Revcontent API using client_credentials
    2. Creating a new campaign (Boost) with targeting criteria
    3. Fetching campaign statistics
    4. Custom exception classes for proper error handling
    5. Comprehensive unit testing with mocked responses

Example:
    Basic usage:
    
        from main import RevcontentAPI, RevcontentConfig, AuthenticationError, CampaignCreationError
        
        config = RevcontentConfig()
        api = RevcontentAPI(config)
        
        try:
            api.authenticate()
            campaign = api.create_campaign("Test Campaign", bid_amount=0.35, budget=50.0)
            stats = api.get_campaign_stats(campaign.id)
        except AuthenticationError as e:
            print(f"Authentication failed: {e}")
        except CampaignCreationError as e:
            print(f"Campaign creation failed: {e}")
"""

import requests
import logging
from typing import Optional, List, Union, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from unittest.mock import Mock, patch
import unittest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RevcontentAPIError(Exception):
    """Base exception for Revcontent API errors."""
    pass


class AuthenticationError(RevcontentAPIError):
    """Raised when authentication fails or is required."""
    pass


class CampaignCreationError(RevcontentAPIError):
    """Raised when campaign creation fails."""
    pass


class StatsRetrievalError(RevcontentAPIError):
    """Raised when statistics retrieval fails."""
    pass


@dataclass
class RevcontentConfig:
    """Configuration for Revcontent API.

    Attributes:
        base_url (str): The base URL for the Revcontent API.
        client_id (str): OAuth2 client ID for authentication.
        client_secret (str): OAuth2 client secret for authentication.
    """
    base_url: str = "https://api.revcontent.io"
    client_id: str = "mock_client_id"
    client_secret: str = "mock_client_secret"


@dataclass
class AuthToken:
    """OAuth2 authentication token response.

    Attributes:
        access_token (str): The bearer token for API requests.
        token_type (str): Type of token (typically 'Bearer').
        expires_in (Optional[int]): Token expiration time in seconds.
        created_at (Optional[datetime]): When the token was created.
    """
    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    created_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Initialize created_at timestamp if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Campaign:
    """Revcontent campaign (Boost) data structure.

    Attributes:
        id (str): Unique campaign identifier.
        name (str): Campaign name.
        bid_amount (float): Bid amount in USD.
        budget (Union[str, float]): Budget ('unlimited' or amount).
        country_codes (Optional[List[str]]): List of ISO country codes.
        tracking_code (Optional[str]): UTM tracking code.
    """
    id: str
    name: str
    bid_amount: float = 0.50
    budget: Union[str, float] = "unlimited"
    country_codes: Optional[List[str]] = None
    tracking_code: Optional[str] = None


@dataclass
class CampaignStats:
    """Campaign statistics data structure.

    Attributes:
        campaign_id (str): Associated campaign ID.
        impressions (int): Number of ad impressions.
        clicks (int): Number of clicks.
        ctr (float): Click-through rate.
        spend (float): Amount spent in USD.
    """
    campaign_id: str
    impressions: int
    clicks: int
    ctr: float
    spend: float


class RevcontentAPI:
    """Revcontent API client for campaign management.

    This class provides methods to authenticate with the Revcontent API,
    create campaigns (Boosts), and retrieve campaign statistics.

    Attributes:
        config (RevcontentConfig): API configuration including credentials.
        auth_token (Optional[AuthToken]): Current authentication token.
        session (requests.Session): HTTP session for making requests.
    """

    def __init__(self, config: RevcontentConfig) -> None:
        """Initialize the Revcontent API client.

        Args:
            config (RevcontentConfig): Configuration object containing API credentials.
        """
        self.config = config
        self.auth_token: Optional[AuthToken] = None
        self.session = requests.Session()

    def authenticate(self) -> bool:
        """Authenticate with the Revcontent API using OAuth2 client credentials.

        Returns:
            bool: True if authentication was successful, False otherwise.

        Raises:
            requests.exceptions.RequestException: If the authentication request fails.
        """
        try:
            auth_url = f"{self.config.base_url}/oauth/token"
            auth_data = {
                "grant_type": "client_credentials",
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
            }

            response = self.session.post(auth_url, data=auth_data)
            response.raise_for_status()

            token_data = response.json()
            access_token = token_data.get("access_token")

            if access_token:
                self.auth_token = AuthToken(
                    access_token=access_token,
                    token_type=token_data.get("token_type", "Bearer"),
                    expires_in=token_data.get("expires_in"),
                )

                self.session.headers.update({
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                })
                logger.info("Authentication successful")
                return True

            return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            return False

    def create_campaign(
        self,
        name: str,
        bid_amount: float = 0.50,
        budget: Union[str, float] = "unlimited",
        country_codes: Optional[List[str]] = None,
        tracking_code: Optional[str] = None,
    ) -> Campaign:
        """Create a new campaign (Boost) with targeting criteria.

        Args:
            name (str): Campaign name (must be unique).
            bid_amount (float): Bid amount in USD (minimum $0.01).
            budget (Union[str, float]): Budget ('unlimited' or amount in USD).
            country_codes (Optional[List[str]]): List of ISO country codes.
            tracking_code (Optional[str]): UTM tracking code.

        Returns:
            Campaign: Campaign object if creation was successful.

        Raises:
            AuthenticationError: If not authenticated.
            CampaignCreationError: If campaign creation fails.
            ValueError: If any parameters are invalid.
        """
        if not self.auth_token:
            logger.error("Not authenticated")
            raise AuthenticationError("Not authenticated. Call authenticate() first.")

        # Basic validation
        if bid_amount < 0.01:
            raise ValueError("Bid amount must be at least $0.01")
        if isinstance(budget, (int, float)) and budget < 1.0:
            raise ValueError("Budget amount must be at least $1.00")

        try:
            campaign_url = f"{self.config.base_url}/stats/api/v1.0/boosts/add"
            campaign_data: Dict[str, Any] = {
                "name": name,
                "bid_amount": str(bid_amount),
                "budget": budget if isinstance(budget, str) else str(budget),
                "country_targeting": "include" if country_codes else "all",
            }

            if country_codes:
                campaign_data["country_codes"] = country_codes
            if tracking_code:
                campaign_data["tracking_code"] = tracking_code

            response = self.session.post(campaign_url, json=campaign_data)
            response.raise_for_status()

            response_data = response.json()
            if response_data.get("success"):
                campaign_id = response_data["data"][0]["id"]
                campaign = Campaign(
                    id=campaign_id,
                    name=name,
                    bid_amount=bid_amount,
                    budget=budget
                )
                logger.info(f"Campaign created successfully: {campaign.id}")
                return campaign
            else:
                error_msg = f"Campaign creation failed: {response_data.get('errors', 'Unknown error')}"
                logger.error(error_msg)
                raise CampaignCreationError(error_msg)

        except requests.exceptions.RequestException as e:
            logger.error(f"Campaign creation failed: {e}")
            raise CampaignCreationError(f"Campaign creation failed: {e}") from e

    def get_campaign_stats(self, campaign_id: str) -> CampaignStats:
        """Fetch statistics for a specific campaign.

        Args:
            campaign_id (str): Unique identifier of the campaign.

        Returns:
            CampaignStats: CampaignStats object if retrieval was successful.

        Raises:
            AuthenticationError: If not authenticated.
            StatsRetrievalError: If statistics retrieval fails.
        """
        if not self.auth_token:
            logger.error("Not authenticated")
            raise AuthenticationError("Not authenticated. Call authenticate() first.")

        try:
            stats_url = f"{self.config.base_url}/stats/boosts/{campaign_id}"

            response = self.session.get(stats_url)
            response.raise_for_status()

            stats_data = response.json()
            stats = CampaignStats(
                campaign_id=campaign_id,
                impressions=stats_data["impressions"],
                clicks=stats_data["clicks"],
                ctr=stats_data["ctr"],
                spend=stats_data["spend"],
            )
            logger.info(f"Stats retrieved for campaign: {campaign_id}")
            return stats

        except requests.exceptions.RequestException as e:
            logger.error(f"Stats retrieval failed: {e}")
            raise StatsRetrievalError(f"Stats retrieval failed: {e}") from e


def print_campaign_results(campaign: Campaign, stats: CampaignStats) -> None:
    """Print campaign and statistics information in a clear, formatted display.

    Args:
        campaign (Campaign): Campaign object containing campaign details.
        stats (CampaignStats): CampaignStats object containing performance metrics.
    """
    print("\n" + "=" * 50)
    print("CAMPAIGN RESULTS")
    print("=" * 50)
    print(f"Campaign ID: {campaign.id}")
    print(f"Campaign Name: {campaign.name}")
    print(f"Bid Amount: ${campaign.bid_amount:.2f}")
    print(f"Budget: {campaign.budget if isinstance(campaign.budget, str) else f'${campaign.budget:.2f}'}")
    if campaign.country_codes:
        print(f"Country Codes: {', '.join(campaign.country_codes)}")
    if campaign.tracking_code:
        print(f"Tracking Code: {campaign.tracking_code}")
    
    print("\nSTATISTICS:")
    print(f"Impressions: {stats.impressions:,}")
    print(f"Clicks: {stats.clicks:,}")
    print(f"CTR: {stats.ctr:.2%}")
    print(f"Spend: ${stats.spend:.2f}")
    print("=" * 50)


def main() -> None:
    """Main execution function that demonstrates the complete Revcontent API workflow."""
    config = RevcontentConfig()
    api = RevcontentAPI(config)

    # Mock the API responses for demonstration
    with patch.object(api.session, "post") as mock_post, \
         patch.object(api.session, "get") as mock_get:

        # Mock authentication response
        mock_auth_response = Mock()
        mock_auth_response.json.return_value = {"access_token": "mock_access_token_123"}
        mock_auth_response.raise_for_status.return_value = None

        # Mock campaign creation response
        mock_campaign_response = Mock()
        mock_campaign_response.json.return_value = {
            "success": True,
            "data": [{"id": "313"}]
        }
        mock_campaign_response.raise_for_status.return_value = None

        # Mock stats response
        mock_stats_response = Mock()
        mock_stats_response.json.return_value = {
            "impressions": 15420,
            "clicks": 231,
            "ctr": 0.015,
            "spend": 23.10,
        }
        mock_stats_response.raise_for_status.return_value = None

        # Configure mock responses
        mock_post.side_effect = [mock_auth_response, mock_campaign_response]
        mock_get.return_value = mock_stats_response

        print("Starting Revcontent Campaign API Integration...")

        # Step 1: Authenticate
        if not api.authenticate():
            print("Authentication failed!")
            return

        # Step 2: Create campaign
        try:
            campaign = api.create_campaign(
                name="Test Campaign - YourName",
                bid_amount=0.35,
                budget=50.00,
                country_codes=["US"],
                tracking_code="utm_source=revcontent"
            )
        except (AuthenticationError, CampaignCreationError, ValueError) as e:
            print(f"Campaign creation failed: {e}")
            return

        # Step 3: Fetch stats
        try:
            stats = api.get_campaign_stats(campaign.id)
        except (AuthenticationError, StatsRetrievalError) as e:
            print(f"Stats retrieval failed: {e}")
            return

        # Step 4: Print results
        print_campaign_results(campaign, stats)


class TestRevcontentAPI(unittest.TestCase):
    """Unit tests for Revcontent API integration."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.config = RevcontentConfig()
        self.api = RevcontentAPI(self.config)

    @patch("requests.Session.post")
    def test_authentication_success(self, mock_post: Mock) -> None:
        """Test successful authentication with valid credentials."""
        mock_response = Mock()
        mock_response.json.return_value = {"access_token": "test_token"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.api.authenticate()

        self.assertTrue(result)
        self.assertIsNotNone(self.api.auth_token)
        if self.api.auth_token:
            self.assertEqual(self.api.auth_token.access_token, "test_token")

    @patch("requests.Session.post")
    def test_campaign_creation(self, mock_post: Mock) -> None:
        """Test campaign creation with valid parameters."""
        self.api.auth_token = AuthToken(access_token="test_token")

        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "data": [{"id": "test_campaign_123"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.api.create_campaign(
            name="Test Campaign", 
            bid_amount=0.35, 
            budget=50.0,
            country_codes=["US"]
        )

        self.assertEqual(result.id, "test_campaign_123")
        self.assertEqual(result.name, "Test Campaign")
        self.assertEqual(result.bid_amount, 0.35)

    @patch("requests.Session.get")
    def test_stats_retrieval(self, mock_get: Mock) -> None:
        """Test statistics retrieval for a campaign."""
        self.api.auth_token = AuthToken(access_token="test_token")

        mock_response = Mock()
        mock_response.json.return_value = {
            "impressions": 1000,
            "clicks": 50,
            "ctr": 0.05,
            "spend": 5.00,
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.api.get_campaign_stats("test_campaign_123")

        self.assertEqual(result.impressions, 1000)
        self.assertEqual(result.clicks, 50)
        self.assertEqual(result.ctr, 0.05)

    def test_campaign_validation(self) -> None:
        """Test campaign parameter validation."""
        self.api.auth_token = AuthToken(access_token="test_token")

        # Test invalid bid amount
        with self.assertRaises(ValueError):
            self.api.create_campaign("Test", bid_amount=0.005)

        # Test invalid budget
        with self.assertRaises(ValueError):
            self.api.create_campaign("Test", budget=0.5)

    def test_authentication_required(self) -> None:
        """Test that methods raise AuthenticationError when not authenticated."""
        # Test campaign creation without authentication
        with self.assertRaises(AuthenticationError):
            self.api.create_campaign("Test Campaign")

        # Test stats retrieval without authentication
        with self.assertRaises(AuthenticationError):
            self.api.get_campaign_stats("test_id")


if __name__ == "__main__":
    # Run the main demo
    main()

    # Run tests
    print("\n" + "=" * 50)
    print("RUNNING UNIT TESTS")
    print("=" * 50)
    unittest.main(argv=[""], exit=False, verbosity=2)