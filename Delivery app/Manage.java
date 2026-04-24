class Manage {
    public void manage(Vehicle v) {
        v.assign();
        System.out.println("the delivery has been assigned");
        v.dispatch();
        System.out.println("the delivery has been dispatched");
        v.delivered();
        System.out.println("the delivery has been delivered successfully");
    }
}